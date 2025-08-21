import io
import json
import requests
import threading
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import torch
from transformers import CLIPProcessor, CLIPModel

MODEL_ID = "clip-model"
clip_model = None
clip_processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def classify(image_bytes: bytes) -> dict:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        texts = [
            "a photo of a newspaper cover with a title and masthead",
            "a photo of an internal page with articles and blocks of body text (not title and masthead)",
            "a photo of an internal page full of advertisements or announcements (not title and masthead)",
            "a photo of an internal page with a large illustration or photograph (not title and masthead)",
            "a photo of a table of contents or an editorial page (not title and masthead)"
        ]
        inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            prob = logits_per_image.softmax(dim=1).cpu().numpy().flatten()
        best = prob.argmax()
        return {
            "prob": float(prob[best]),
            "is_cover": bool(best == 0)
        }
    except Exception as e:
        return {"error": f"Błąd przetwarzania obrazu: {e}"}

class ManifestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Struktura gazety")
        self.root.geometry("800x650")

        self.frame = ttk.Frame(root, padding="10")
        self.frame.pack(fill=tk.X)

        ttk.Label(self.frame, text="Link do manifestu IIIF:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.url_entry = ttk.Entry(self.frame, width=80)
        self.url_entry.grid(row=0, column=1, sticky=tk.EW, columnspan=2)
        
        self.button1 = ttk.Button(self.frame, text="Pobierz informacje", command=self.start_fetch)
        self.button1.grid(row=0, column=3, padx=5)
        
        analysis_options_frame = ttk.Frame(self.frame)
        analysis_options_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=5)

        ttk.Label(analysis_options_frame, text="Zakres stron:").pack(side=tk.LEFT, padx=(0, 5))
        self.start_entry = ttk.Entry(analysis_options_frame, width=10, state=tk.DISABLED)
        self.start_entry.pack(side=tk.LEFT)
        ttk.Label(analysis_options_frame, text="–").pack(side=tk.LEFT, padx=5)
        self.end_entry = ttk.Entry(analysis_options_frame, width=10, state=tk.DISABLED)
        self.end_entry.pack(side=tk.LEFT)

        self.is_split_scan_var = tk.BooleanVar(value=False)
        self.split_scan_check = ttk.Checkbutton(analysis_options_frame, text="Skan dwustronicowy (dziel na pół)", variable=self.is_split_scan_var, state=tk.DISABLED)
        self.split_scan_check.pack(side=tk.LEFT, padx=(20, 0))

        self.button2 = ttk.Button(self.frame, text="Rozpocznij Analizę", command=self.start_search, state=tk.DISABLED)
        self.button2.grid(row=1, column=3, padx=5)

        self.button3 = ttk.Button(self.frame, text="Edytuj i Zapisz Manifest", command=self.open_editor, state=tk.DISABLED)
        self.button3.grid(row=2, column=1, columnspan=3, sticky=tk.E, pady=5)

        self.progress_frame = ttk.Frame(root, padding="0 10 10 10")
        self.progress_frame.pack(fill=tk.X)
        self.progress_label = ttk.Label(self.progress_frame, text="Postęp:")
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient='horizontal', mode='determinate')
        self.progress_percent = ttk.Label(self.progress_frame, text="0%")

        self.log_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=35)
        self.log_box.pack(pady=(0, 10), padx=10, fill=tk.BOTH, expand=True)

        self.frame.columnconfigure(1, weight=1)
        self.canvases = []
        self.total_pages = 0
        self.manifest = None
        self.analysis_results = []

        self.log("Wklej link do manifestu, a następnie kliknij 'Pobierz informacje'.")
        self.log("Jeśli skany zawierają po dwie strony, zaznacz opcję 'Skan dwustronicowy'.")

    def log(self, message):
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.see(tk.END)

    def toggle_ui(self, state):
        is_manifest_loaded = self.total_pages > 0
        
        self.button1.config(state=state)
        self.url_entry.config(state=state)
        
        self.button2.config(state=tk.DISABLED if state == tk.DISABLED or not is_manifest_loaded else tk.NORMAL)
        self.start_entry.config(state=tk.DISABLED if state == tk.DISABLED or not is_manifest_loaded else tk.NORMAL)
        self.end_entry.config(state=tk.DISABLED if state == tk.DISABLED or not is_manifest_loaded else tk.NORMAL)
        self.split_scan_check.config(state=tk.DISABLED if state == tk.DISABLED or not is_manifest_loaded else tk.NORMAL)

        is_analysis_done = bool(self.analysis_results)
        self.button3.config(state=tk.DISABLED if state == tk.DISABLED or not is_analysis_done else tk.NORMAL)

    def toggle_progress_bar(self, show=True):
        if show:
            self.progress_label.grid(row=0, column=0, padx=(0, 5))
            self.progress_bar.grid(row=0, column=1, sticky=tk.EW)
            self.progress_percent.grid(row=0, column=2, padx=(5, 0))
            self.progress_frame.columnconfigure(1, weight=1)
        else:
            for widget in self.progress_frame.winfo_children():
                widget.grid_remove()

    def start_fetch(self):
        self.toggle_ui(tk.DISABLED)
        self.analysis_results = []
        threading.Thread(target=self.fetch_manifest, daemon=True).start()

    def fetch_manifest(self):
        url = self.url_entry.get().strip()
        if not url:
            self.log("Błąd: Pole z linkiem do manifestu jest puste.")
            self.root.after(0, self.toggle_ui, tk.NORMAL)
            return

        try:
            self.log(f"\nPobieranie informacji z manifestu: {url}")
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            self.manifest = response.json()
            self.canvases = self.manifest.get('sequences', [{}])[0].get('canvases', [])
            self.total_pages = len(self.canvases)

            if self.total_pages == 0:
                self.log("Błąd: W podanym manifeście nie znaleziono żadnych stron.")
            else:
                self.log(f"Znaleziono {self.total_pages} stron.")
                self.start_entry.delete(0, tk.END)
                self.start_entry.insert(0, "1")
                self.end_entry.delete(0, tk.END)
                self.end_entry.insert(0, str(self.total_pages))
        except Exception as e:
            self.log(f"Błąd podczas pobierania manifestu: {e}")
            self.manifest = None
            self.total_pages = 0
        finally:
            self.root.after(0, self.toggle_ui, tk.NORMAL)

    def start_search(self):
        try:
            start_page = int(self.start_entry.get())
            end_page = int(self.end_entry.get())
            
            if not (1 <= start_page <= self.total_pages):
                self.log(f"Błąd: 'Od' musi być liczbą od 1 do {self.total_pages}.")
                return
            if not (1 <= end_page <= self.total_pages):
                self.log(f"Błąd: 'Do' musi być liczbą od 1 do {self.total_pages}.")
                return
            if start_page > end_page:
                self.log("Błąd: Strona początkowa ('Od') nie może być większa niż końcowa ('Do').")
                return
        except ValueError:
            self.log("Błąd: Wprowadź poprawne liczby w polach zakresu stron.")
            return

        self.toggle_ui(tk.DISABLED)
        self.toggle_progress_bar(True)
        self.progress_bar['value'] = 0
        self.progress_percent.config(text="0%")
        threading.Thread(target=self.run_search, args=(start_page, end_page), daemon=True).start()

    def run_search(self, start_page, end_page):
        is_split_scan = self.is_split_scan_var.get()
        self.log("\n" + "="*80)
        self.log(f"Rozpoczynam analizę stron od {start_page} do {end_page}")
        if is_split_scan:
            self.log("Tryb: Skan dwustronicowy (obrazy będą dzielone na pół)")

        start_index = start_page - 1
        end_index = end_page
        canvases_to_analyze = self.canvases[start_index:end_index]
        total_to_process = len(canvases_to_analyze)
        
        self.analysis_results = []
        for i, canvas in enumerate(canvases_to_analyze):
            page_num = start_page + i
            
            page_data = {
                "id_text": f"Strona {page_num}",
                "page_num": page_num,
                "canvas_id": canvas.get('@id'),
                "is_cover": False,
                "prob": 0.0
            }
            
            image_url_base = canvas.get('images', [{}])[0].get('resource', {}).get('service', {}).get('@id')
            if not image_url_base:
                self.log(f"Strona {page_num}: Brak adresu URL obrazu. Pomijam.")
                self.analysis_results.append(page_data)
                continue

            try:
                full_image_url = f"{image_url_base.rstrip('/')}/full/1200,/0/default.jpg"
                response = requests.get(full_image_url, timeout=30)
                response.raise_for_status()
                
                if not is_split_scan:
                    result = classify(response.content)
                    if 'error' not in result:
                        page_data.update(result)
                    else:
                        self.log(f"Błąd analizy strony {page_num}: {result['error']}")
                else:
                    image = Image.open(io.BytesIO(response.content))
                    width, height = image.size
                    mid_point = width // 2
                    
                    left_img_pil = image.crop((0, 0, mid_point, height))
                    left_bytes_io = io.BytesIO()
                    left_img_pil.save(left_bytes_io, format='JPEG')
                    result_left = classify(left_bytes_io.getvalue())

                    right_img_pil = image.crop((mid_point, 0, width, height))
                    right_bytes_io = io.BytesIO()
                    right_img_pil.save(right_bytes_io, format='JPEG')
                    result_right = classify(right_bytes_io.getvalue())

                    is_left_cover = result_left.get('is_cover', False)
                    is_right_cover = result_right.get('is_cover', False)

                    page_data['is_cover'] = is_left_cover or is_right_cover
                    page_data['prob'] = max(result_left.get('prob', 0), result_right.get('prob', 0))

            except Exception as e:
                self.log(f"Błąd pobierania lub przetwarzania strony {page_num}: {e}")
            
            self.analysis_results.append(page_data)
            
            progress = (i + 1) / total_to_process * 100
            self.root.after(0, self.update_progress, progress)

        self.root.after(0, self.show_summary)

    def update_progress(self, value):
        self.progress_bar['value'] = value
        self.progress_percent.config(text=f"{int(value)}%")

    def show_summary(self):
        self.toggle_progress_bar(False)
        self.log("\n### PODSUMOWANIE ANALIZY ###")

        covers = [p for p in self.analysis_results if p.get("is_cover")]

        if not covers:
            self.log("\nNie zidentyfikowano żadnej strony jako okładki w podanym zakresie.")
        else:
            self.log(f"\nZnaleziono {len(covers)} potencjalnych okładek:")
            covers.sort(key=lambda x: x['page_num'])
            for cover in covers:
                prob_str = f"{cover.get('prob', 0):.2%}"
                self.log(f"- Strona {cover['page_num']:<10} | Prawdopodobieństwo: {prob_str}")

        self.log("\n" + "#"*80)
        self.log("Analiza zakończona. Możesz teraz edytować wyniki i zapisać manifest.")
        self.toggle_ui(tk.NORMAL)

    def open_editor(self):
        if not self.manifest:
            self.log("Błąd: Najpierw załaduj manifest.")
            return

        editor_win = tk.Toplevel(self.root)
        editor_win.title("Edycja Struktury")
        editor_win.geometry("600x600")
        editor_win.transient(self.root)
        editor_win.grab_set()

        main_frame = ttk.Frame(editor_win, padding=10)
        main_frame.pack(fill="both", expand=True)

        columns = ('check', 'label')
        tree = ttk.Treeview(main_frame, columns=columns, show='headings', selectmode='none')
        
        tree.heading('check', text='')
        tree.heading('label', text='Strona')
        tree.column('check', width=50, stretch=tk.NO, anchor='center')
        tree.column('label', stretch=tk.YES)

        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        self.check_state = {}
        results_map = {p['page_num']: p for p in self.analysis_results}

        unchecked_char = "☐"
        checked_char = "☑"

        for i in range(self.total_pages):
            page_num = i + 1
            result_data = results_map.get(page_num)
            is_cover = result_data.get("is_cover", False) if result_data else False
            self.check_state[page_num] = is_cover
            
            char = checked_char if is_cover else unchecked_char
            label = f"Strona {page_num}"
            tree.insert('', tk.END, iid=str(page_num), values=(char, label))
            
        def toggle_check(event):
            item_id_str = tree.identify_row(event.y)
            column = tree.identify_column(event.x)
            if not item_id_str or column != '#1':
                return

            page_num = int(item_id_str)
            self.check_state[page_num] = not self.check_state[page_num]
            
            new_char = checked_char if self.check_state[page_num] else unchecked_char
            current_values = tree.item(item_id_str, 'values')
            tree.item(item_id_str, values=(new_char, current_values[1]))
            
        tree.bind('<Button-1>', toggle_check)

        button_frame = ttk.Frame(editor_win, padding=10)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM)

        def deselect_all():
            for page_num in self.check_state:
                if self.check_state[page_num]:
                    self.check_state[page_num] = False
                    current_values = tree.item(str(page_num), 'values')
                    tree.item(str(page_num), values=(unchecked_char, current_values[1]))

        deselect_button = ttk.Button(button_frame, text="Odznacz wszystkie", command=deselect_all)
        deselect_button.pack(side=tk.LEFT, padx=5)
        
        save_button = ttk.Button(
            button_frame, text="Zapisz manifest.json",
            command=lambda: self.save_manifest(self.check_state, editor_win)
        )
        save_button.pack(side=tk.RIGHT, padx=5)

    def save_manifest(self, check_state, editor_win):
        if not self.manifest:
            self.log("Błąd: Brak danych manifestu do zapisu.")
            editor_win.destroy()
            return
        
        cover_pages = [num for num, checked in check_state.items() if checked]
        cover_pages.sort()

        if not cover_pages:
            self.log("Nie zaznaczono żadnych okładek. Zapisuję manifest bez pola 'structures'.")
            self.manifest.pop('structures', None)
        else:
            self.log(f"\nWybrane okładki to strony: {cover_pages}")
            self.log("Generowanie nowej struktury manifestu.")
            
            base_id = self.manifest.get('@id', 'http://example.com/manifest')
            if not base_id or not base_id.strip():
                base_id = 'http://example.com/manifest'

            structures = []
            for i, start_page in enumerate(cover_pages):
                label = f"Wydanie rozpoczynające się od strony {start_page}"
                range_id = f"{base_id.rstrip('/')}/range/r{i}"
                
                end_page = cover_pages[i+1] - 1 if i + 1 < len(cover_pages) else self.total_pages
                
                start_index = start_page - 1
                end_index = end_page 
                
                range_canvas_ids = [c['@id'] for c in self.canvases[start_index:end_index] if '@id' in c]

                if range_canvas_ids:
                    structures.append({
                        "@id": range_id,
                        "@type": "sc:Range",
                        "label": label,
                        "canvases": range_canvas_ids
                    })
            
            self.manifest['structures'] = structures
            self.log("Struktura została wygenerowana.")

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="manifest.json",
            title="Zapisz manifest jako..."
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.manifest, f, indent=4, ensure_ascii=False)
                self.log(f"Manifest został pomyślnie zapisany w: {file_path}")
            except Exception as e:
                self.log(f"Błąd: Nie udało się zapisać pliku. Szczegóły: {e}")
                messagebox.showerror("Błąd zapisu", f"Nie udało się zapisać pliku:\n{e}")
        else:
            self.log("Zapis anulowany przez użytkownika.")

        editor_win.destroy()

if __name__ == "__main__":
    MODEL_PATH = "clip-model"
    
    print(f"Ładowanie modelu: {MODEL_PATH}")
    
    try:
        clip_model = CLIPModel.from_pretrained(MODEL_PATH).to(device)
        clip_processor = CLIPProcessor.from_pretrained(MODEL_PATH)
        print(f"\nModel został załadowany i działa na: {device.upper()}")

        root = tk.Tk()
        app = ManifestApp(root)
        root.mainloop()
        
    except Exception as e:
        error_message = (
            f"Błąd: Nie udało się załadować modelu z folderu '{MODEL_PATH}'.\n\n"
            f"Szczegóły błędu:\n{e}"
        )
        print(error_message)
        
        error_root = tk.Tk()
        error_root.withdraw()
        messagebox.showerror("Błąd ładowania modelu", error_message)
        error_root.destroy()
        input("\nNaciśnij Enter, aby zamknąć program.")