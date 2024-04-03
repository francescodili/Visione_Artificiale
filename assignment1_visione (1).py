import cv2
import numpy as np
#Libreria per gestione eventi da tastiera
import keyboard
#Librerie per creazione e personalizzazione finestre
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


#Funzione per estrarre un frame equirettangolare da un video a 360
def extract_frame_equirectangular(video_path, frame_number):
    # Apertura del video
    cap = cv2.VideoCapture(video_path)
    # Impostazione del frame da leggere 
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    # Lettura del frame
    ret, frame = cap.read() #return_value (bool), image
    #Chiude il video(non ci serve più dopo aver preso il frame)
    cap.release()
    if not ret:
        print("Errore nella lettura del frame.")
        return

    return frame

#Funzione per passare l'immagine equirettangolare(coordinate sferiche) in immagine prospettica(cooridnate cartesiane)
def equirectangular_to_perspective(equirectangular, fov, theta_degree, phi_degree, height_out, width_out):
    # Converti gli angoli di campo visivo e orientamento da gradi a radianti
    theta_rad = np.radians(theta_degree)
    phi_rad = np.radians(phi_degree)
    fov_rad = np.radians(fov)

    # Calcola i fattori di scala basati sulle dimensioni dell'immagine equirettangolare
    w_ratio = equirectangular.shape[1] / (2 * np.pi)
    h_ratio = equirectangular.shape[0] / np.pi

    # Griglia di coordinate per l'immagine di output
    z_out, y_out = np.meshgrid(np.arange(width_out), np.arange(height_out))

    # Normalizza le coordinate (da 0 a 1) e applica il FOV come proiettato sul piano tangente alla sfera
    z_out = ((z_out / width_out) - 0.5) * 2 * np.tan(fov_rad / 2)
    y_out = -((y_out / height_out) - 0.5) * 2 * np.tan(fov_rad / 2)

    # Calcola il punto P sul piano tangente canonico (x=1)
    P = np.stack((np.ones_like(y_out), y_out, z_out), axis=-1)

    # Normalizza P per proiettarlo sulla sfera unitaria
    P /= np.linalg.norm(P, axis=2, keepdims=True)

    # Matrice di rotazione per theta (rotazione attorno all'asse y)
    R_theta = np.array([
        [np.cos(theta_rad), 0, np.sin(theta_rad)],
        [0, 1, 0],
        [-np.sin(theta_rad), 0, np.cos(theta_rad)]
    ])

    # Matrice di rotazione per phi (rotazione attorno all'asse z)
    R_phi = np.array([
        [np.cos(phi_rad), np.sin(phi_rad), 0],
        [-np.sin(phi_rad), np.cos(phi_rad), 0],
        [0, 0, 1]
    ])

    # Crea la matrice di rotazione combinata, prima phi e poi theta facendo la moltiplicazione tra matrici
    R = R_phi @ R_theta
    #Applico la rotazione combinata a ogni punto P
    #Moltiplicazione tra la matrice multidimensionale P e la matrice di rotazione R
    #ijk dimensione di P, kl dimensione di R, ijl dimensione di P_rotated. (ij da P, l da R)
    P_rotated = np.einsum('ijk,kl->ijl', P, R)

    # Calcola le coordinate equirettangolari equivalenti
    x_eq = w_ratio * (np.arctan2(P_rotated[..., 2], P_rotated[..., 0]) + np.pi)
    y_eq = h_ratio * ((np.pi / 2) - np.arcsin(P_rotated[..., 1]))

    # Faccio il cast dei valori in float32 per la funzione remap
    map_x_32 = x_eq.astype('float32')
    map_y_32 = y_eq.astype('float32')

    # Interpolazione bilineare per mappare i pixel
    perspective_image = cv2.remap(equirectangular, map_x_32, map_y_32, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    return perspective_image


#Funzione per aggiornare l'immagine mostrata nella finestra
def update_image():
    global theta, phi, fov, equirectangular, height_out, width_out
    if equirectangular is not None:  # Assicura che equirectangular sia definita
        perspective_image = equirectangular_to_perspective(equirectangular, fov, theta, phi, height_out, width_out)
        if perspective_image is not None:
            show_image_on_label(perspective_image)
            update_status_label()
        else:
            print("Errore nell'aggiornamento dell'immagine.")

def show_image_on_label(img):
    try:
        # Converti l'immagine BGR di OpenCV in formato RGB per PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Crea un'immagine PIL dall'array
        pil_image = Image.fromarray(img_rgb)
        # Converte l'immagine PIL in un formato che Tkinter può usare
        tk_image = ImageTk.PhotoImage(image=pil_image)
        # Aggiorna l'immagine sul label
        image_label.image = tk_image
        image_label['image'] = image_label.image
    except tk.TclError:
        pass



def update_video_label(frame):
    # Converti l'immagine BGR di OpenCV in formato RGB per PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    tk_image = ImageTk.PhotoImage(image=pil_image)

    # Aggiorna l'immagine su video_label
    video_label.configure(image=tk_image)
    video_label.image = tk_image  # Evita la garbage collection

def update_frame_number_slider(frame_number, total_frames):
    # Calcola la posizione dello slider basata sul numero di frame corrente
    slider_position = (frame_number / total_frames) * 100
    frame_number_slider.set(slider_position)

def show_transformed_video(video_path):
    global cap
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Errore nell'apertura del video")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame():
        ret, frame = cap.read()
        if ret:
            # Ottiene il numero del frame corrente
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            # Applica la trasformazione a ogni frame qui
            transformed_frame = equirectangular_to_perspective(frame, fov, theta, phi, height_out, width_out)
            update_video_label(transformed_frame)
            update_frame_label(current_frame)
            update_status_label()
            
            # Aggiorna la posizione dello slider per il numero di frame
            update_frame_number_slider(current_frame, total_frames)
            
            # Pianifica il prossimo frame
            window_video.after(16, read_frame) 
        else:
            cap.release()

    read_frame()

def update_with_input(*args):
    global theta, phi, fov, equirectangular, video_paths
    video_path = video_paths.get()  # Ottiene il percorso del video selezionato
    try:
        # Prova a convertire i valori inseriti dall'utente da stringa a intero
        theta_temp = int(theta_entry.get())
        phi_temp = int(phi_entry.get())
        fov_temp = int(fov_entry.get())

        # Aggiorna le variabili globali con i valori validi
        theta = normalize_theta(theta_temp)
        phi = normalize_phi(phi_temp)
        fov = normalize_fov(fov_temp)
        
        # Aggiorna gli slider e le etichette con i valori reimpostati
        theta_slider.set(theta)
        phi_slider.set(phi)
        fov_slider.set(fov)

        # Estrae nuovamente il frame equirettangolare usando il percorso del video aggiornato
        equirectangular = extract_frame_equirectangular(video_path, frame_number)
        update_image()
    except ValueError:
        # Gestisce il caso in cui la conversione fallisca
        print("Inserire valori interi validi per Theta, Phi e FOV.")

def update_video_with_input(*args):
    global theta, phi, fov, video_paths, frame_number, cap
    # Assicurati di fermare il loop di lettura del frame corrente e rilasciare la risorsa
    if 'cap' in globals() and cap.isOpened():
        cap.release()

    # Aggiorna il percorso del video selezionato
    video_path = video_paths.get()
    
    # Reimposta i valori di theta, phi e fov
    theta = 0
    phi = 0
    fov = 70

    # Imposta il frame iniziale a 0
    frame_number = 0

    # Resetta lo slider del numero di frame, se necessario
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number_slider.configure(from_=0, to=total_frames-1)
    frame_number_slider.set(0)

    # Aggiorna gli slider e le etichette con i valori reimpostati
    if 'theta_slider' in globals():
        theta_slider.set(theta)
    if 'phi_slider' in globals():
        phi_slider.set(phi)
    if 'fov_slider' in globals():
        fov_slider.set(fov)
    
    update_status_label()

    try:
        # Avvia la visualizzazione del nuovo video
        show_transformed_video(video_path)
    except ValueError:
        print("Errore nel cambio del video")
    except Exception as e:
        print(f"Errore imprevisto: {e}")




def update_status_label():
    # Aggiorna il testo del label con i valori correnti di theta, phi e fov
    status_text = f"Theta: {int(theta)}°, Phi: {int(phi)}°, FOV: {int(fov)}°"
    status_label.config(text=status_text)

def update_frame_label(n_frame):
    # Aggiorna il testo del label con il valore corrente del frame
    frame_text = f"{n_frame}"
    number_of_frame_label.config(text=frame_text)


# Setup iniziale
video_path = 'video_1.MP4'
frame_number = 170 #un frame randome prelevato dal video
# Angoli di rotazione in gradi (theta attorno all'asse verticale (longitudine), Phi l'inclinazione su e giù (latitudine))
theta = 0
phi = 0
fov = 70  # Campo visivo in gradi (field of view)
height_out = 500  # Altezza dell'immagine di output
aspect_ratio = 16 / 9 # Aspect Ratio per le proporzioni
width_out = int(height_out * aspect_ratio)   # Larghezza dell'immagine di output
equirectangular = None 


def setup_main_window():
    window_main = tk.Tk()
    window_main.title("Selezione Browsing")

    # Impostazioni per la dimensione e posizione della finestra principale
    window_main.geometry("300x120")  # Larghezza x Altezza
    window_main.resizable(False, False)  # Impedisce il ridimensionamento della finestra

    # Creazione di un frame centrale per contenere i pulsanti
    main_frame = tk.Frame(window_main)
    main_frame.pack(expand=True, fill="both")

    # Pulsante per il browsing di immagini
    img_button = tk.Button(main_frame, text="Browsing Immagine", command=initialize_and_show_image_window)
    img_button.pack(pady=10, expand=True)

    # Pulsante per il browsing di video
    video_button = tk.Button(main_frame, text="Browsing Video", command=initialize_and_show_selector_video_window)
    video_button.pack(pady=10, expand=True)

    # Centratura dei widget nel frame
    main_frame.pack(anchor="center")

    window_main.mainloop()



def change_theme():
        global current_theme_index  # Riferimento alla variabile globale per aggiornarla
        # Calcola il prossimo indice, assicurandosi di rimanere all'interno della lista
        current_theme_index = (current_theme_index + 1) % len(themes)
        # Ottiene il nome del prossimo tema dalla lista
        next_theme = themes[current_theme_index]
        # Imposta il nuovo tema
        style.theme_use(next_theme)

# Funzioni per aggiornare Theta, Phi e FOV quando gli slider vengono mossi
def update_theta(value):
    global theta
    theta = int(float(value))
    update_image()

def update_phi(value):
    global phi
    phi = int(float(value))
    update_image()

def update_fov(value):
    global fov
    fov = int(float(value))
    update_image()

# Definire la funzione per inizializzare e mostrare la finestra 'window'
def initialize_and_show_image_window():
    global window, image_label, status_label, theta_slider, phi_slider, fov_slider, video_paths, frame_number, equirectangular, theta_entry, phi_entry, fov_entry, current_theme_index, style, themes
    
    # Creazione e impostazione della finestra con libreria tkinter
    window = tk.Toplevel()
    window.update_idletasks()  # Aggiorna la finestra
    width = window.winfo_width()
    height = window.winfo_height()
    # Imposta la dimensione minima della finestra alle sue dimensioni correnti
    window.minsize(width, height)
    window.title("Visualizzatore immagine prospettica")
    window.geometry('+3+3') 

    # Lega gli eventi di pressione e rilascio dei tasti alla finestra
    window.bind('<KeyPress>', on_key_press)
    window.bind('<KeyRelease>', on_key_release)
    keyboard.on_press_key("enter", on_enter)

    style = ttk.Style()

    #Frame contenitore dei tre elementi della finestra (informazioni, immagine e input)
    frm = ttk.Frame(window)
    frm.pack(expand=True)  # expand=True permette a frm di espandersi al centro della finestra

    #Frame di informazioni contenente le coordinate attuali dall'img equirettangolare e le istruzioni per l'uso
    info_frame = ttk.Frame(frm)
    info_frame.pack()

    #Label coordinate
    status_label = ttk.Label(info_frame, text="")
    status_label.pack(side=tk.LEFT, padx=10)

    instructions_text = """Istruzioni:
    - Usa le frecce destra/sinistra per spostarti orizzontalmente(theta).
    - Usa le frecce su/giù per spostarti verticalmente(phi).
    - Usa i tasti '+' e '-' per aumentare o diminuire il FOV di 5°."""

    #Label istruzioni
    instructions_label = ttk.Label(info_frame, text=instructions_text)
    instructions_label.pack(side=tk.LEFT, padx=10)

    #Label per l'immagine (inizialmente vuoto)
    image_label = ttk.Label(frm)
    image_label.pack()

    #Frame con le opzioni di input
    inputs_frame = ttk.Frame(frm, padding=10)
    inputs_frame.pack()

    #DropdownList per selezionare da quale video prelevare il frame
    video_paths = tk.StringVar(value="video_1.MP4")
    video_options = ["video_1.MP4", "video_2.MP4"]
    video_dropdown = tk.OptionMenu(inputs_frame, video_paths, *video_options, command=update_with_input)
    video_dropdown.grid(column=0, row=0)

    # Label ed entry per la modifica di Theta
    theta_label = tk.Label(inputs_frame, text="Theta:").grid(column=1, row=0)
    theta_entry = tk.Entry(inputs_frame)
    theta_entry.grid(column=2, row=0)
    theta_entry.insert(0, theta)

    # Label ed entry per la modifica di Phi
    phi_label = tk.Label(inputs_frame, text="Phi:").grid(column=3, row=0)
    phi_entry = tk.Entry(inputs_frame)
    phi_entry.grid(column=4, row=0)
    phi_entry.insert(0, phi)

    # Label ed entry per la modifica di Fov
    fov_label = tk.Label(inputs_frame, text="Fov:").grid(column=5, row=0)
    fov_entry = tk.Entry(inputs_frame)
    fov_entry.grid(column=6, row=0)
    fov_entry.insert(0, fov)

    # Bottone per aggiornare l'immagine con l'input (attivabile anche con ENTER)
    update_button = tk.Button(inputs_frame, text="Aggiorna Vista", command=update_with_input)
    update_button.grid(column=7, row=0)

    # Lista dei temi disponibili
    themes = ['winnative', 'clam', 'alt', 'default', 'classic', 'vista', 'xpnative']
    # Indice iniziale
    current_theme_index = 0

    # Crea un pulsante per cambiare il tema
    theme_button = tk.Button(inputs_frame, text="Cambia Tema", command=change_theme, fg='black', bg='white', activebackground='red' )#colore attivazione pulsante tipo hover quando fai il click nelle pagine
    theme_button.grid(column=8, row=0)


    # Estrai il frame equirettangolare
    equirectangular = extract_frame_equirectangular(video_paths.get(), frame_number)


    # Crea un frame per contenere gli slider
    slider_frame = ttk.Frame(window)
    slider_frame.pack(fill='both', expand=True, padx=10, pady=10)

    # Crea un'etichetta e uno slider per Theta
    theta_label = ttk.Label(slider_frame, text="Dx e Sx (Theta):", anchor='w')
    theta_label.grid(row=0, column=0, sticky='nsew')
    theta_slider = ttk.Scale(slider_frame, from_=-180, to=180, length=400, orient='horizontal', command=update_theta)
    theta_slider.set(theta)  # Imposta il valore iniziale
    theta_slider.grid(row=1, column=0, sticky='nsew')

    # Crea un'etichetta e uno slider per Phi
    phi_label = ttk.Label(slider_frame, text="Su e Giù (Phi):", anchor='w')
    phi_label.grid(row=2, column=0, sticky='nsew')
    phi_slider = ttk.Scale(slider_frame, from_=-90, to=90, length=400, orient='horizontal', command=update_phi)
    phi_slider.set(phi)  # Imposta il valore iniziale
    phi_slider.grid(row=3, column=0, sticky='nsew')

    # Crea un'etichetta e uno slider per FOV
    fov_label = ttk.Label(slider_frame, text="Zoom (FOV):", anchor='w')
    fov_label.grid(row=4, column=0, sticky='nsew')
    fov_slider = ttk.Scale(slider_frame, from_=5, to=90, length=400, orient='horizontal', command=update_fov)
    fov_slider.set(fov)  # Imposta il valore iniziale
    fov_slider.grid(row=5, column=0, sticky='nsew')

    # Imposta le opzioni di espansione della griglia
    slider_frame.grid_rowconfigure(0, weight=1)
    slider_frame.grid_rowconfigure(1, weight=1)
    slider_frame.grid_rowconfigure(2, weight=1)
    slider_frame.grid_rowconfigure(3, weight=1)
    slider_frame.grid_rowconfigure(4, weight=1)
    slider_frame.grid_rowconfigure(5, weight=1)
    slider_frame.grid_columnconfigure(0, weight=1)
        

def initialize_and_show_selector_video_window():
    global video_paths, window_selection
    # Creazione della finestra secondaria
    window_selection = tk.Toplevel()
    window_selection.title("Seleziona Video")
    
    # Impostazioni per la dimensione e posizione della finestra
    window_selection.geometry("300x100")  # Larghezza x Altezza
    window_selection.resizable(False, False)  # Impedisce il ridimensionamento della finestra

    # Creazione di un frame centrale per contenere gli elementi
    main_frame = tk.Frame(window_selection)
    main_frame.pack(expand=True, fill="both")

    # Creazione e posizionamento del menu a tendina per la selezione del video
    video_paths = tk.StringVar(value="video_1.MP4")
    video_options = ["video_1.MP4", "video_2.MP4"]
    video_dropdown = tk.OptionMenu(main_frame, video_paths, *video_options)
    video_dropdown.pack(side="left", padx=10, pady=10, expand=True)

    # Pulsante per confermare la selezione del video
    select_button = tk.Button(main_frame, text="Seleziona", command=select_video)
    select_button.pack(side="left", padx=10, pady=10, expand=True)

    # Centratura dei widget nel frame
    main_frame.pack(anchor="center")

    

def select_video():
    global video_selected, window_video

    # Ottiene il percorso del video selezionato dal dropdown
    video_selected = video_paths.get()

    # Chiude la finestra corrente se è aperta
    if 'window_selection' in globals():
        window_selection.destroy()

    # Chiama la funzione per inizializzare e mostrare la finestra del video con il video selezionato
    initialize_and_show_video_window()


# Funzione chiamata dal pulsante di browsing video
def initialize_and_show_video_window():
    global window_video, video_label, status_label, theta_slider, phi_slider, fov_slider, video_paths, frame_number, equirectangular, theta_entry, phi_entry, fov_entry, current_theme_index, style, themes, frame_number_slider, number_of_frame_label
    
    window_video = tk.Toplevel()
    window_video.update_idletasks()
    width = window_video.winfo_width()
    height = window_video.winfo_height()
    window_video.minsize(width, height)
    window_video.title("Visualizzatore video prospettico")
    window_video.geometry('+3+3')

    window_video.bind('<KeyPress>', on_key_press)
    window_video.bind('<KeyRelease>', on_key_release)

    style = ttk.Style()

    frm = ttk.Frame(window_video)
    frm.pack(expand=True)

    info_frame = ttk.Frame(frm)
    info_frame.pack()

    status_label = ttk.Label(info_frame, text="")
    status_label.pack(side=tk.LEFT, padx=10)

    instructions_text = """Istruzioni:
        - Usa le frecce destra/sinistra per spostarti orizzontalmente(theta).
        - Usa le frecce su/giù per spostarti verticalmente(phi).
        - Usa i tasti '+' e '-' per aumentare o diminuire il FOV di 5°."""

    instructions_label = ttk.Label(info_frame, text=instructions_text)
    instructions_label.pack(side=tk.LEFT, padx=10)

    video_label = ttk.Label(frm)
    video_label.pack()


    cap = cv2.VideoCapture(video_selected)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Crea un'etichetta e uno slider per il numero del frame
    frame_number_label = ttk.Label(frm, text="Frame:", anchor='w')
    frame_number_label.pack(side=tk.LEFT, padx=10)
    frame_number_slider = ttk.Scale(frm, from_=0, to=total_frames, orient='horizontal', length=400)
    frame_number_slider.pack(side=tk.LEFT, padx=10)
    number_of_frame_label = ttk.Label(frm, text="", anchor='w')
    number_of_frame_label.pack(side=tk.LEFT, padx=10)


    inputs_frame = ttk.Frame(frm, padding=10)
    inputs_frame.pack()


    # Lista dei temi disponibili
    themes = ['winnative', 'clam', 'alt', 'default', 'classic', 'vista', 'xpnative']
    # Indice iniziale
    current_theme_index = 0

    theme_button = tk.Button(inputs_frame, text="Cambia Tema", command=change_theme, fg='black', bg='white', activebackground='red')
    theme_button.grid(column=8, row=0)

    slider_frame = ttk.Frame(window_video)
    slider_frame.pack(fill='both', expand=True, padx=10, pady=10)

    theta_label = ttk.Label(slider_frame, text="Dx e Sx (Theta):", anchor='w')
    theta_label.grid(row=0, column=0, sticky='nsew')
    theta_slider = ttk.Scale(slider_frame, from_=-180, to=180, length=400, orient='horizontal', command=update_theta)
    theta_slider.set(theta)  # Imposta il valore iniziale
    theta_slider.grid(row=1, column=0, sticky='nsew')

    # Crea un'etichetta e uno slider per Phi
    phi_label = ttk.Label(slider_frame, text="Su e Giù (Phi):", anchor='w')
    phi_label.grid(row=2, column=0, sticky='nsew')
    phi_slider = ttk.Scale(slider_frame, from_=-90, to=90, length=400, orient='horizontal', command=update_phi)
    phi_slider.set(phi)  # Imposta il valore iniziale
    phi_slider.grid(row=3, column=0, sticky='nsew')

    # Crea un'etichetta e uno slider per FOV
    fov_label = ttk.Label(slider_frame, text="Zoom (FOV):", anchor='w')
    fov_label.grid(row=4, column=0, sticky='nsew')
    fov_slider = ttk.Scale(slider_frame, from_=5, to=90, length=400, orient='horizontal', command=update_fov)
    fov_slider.set(fov)  # Imposta il valore iniziale
    fov_slider.grid(row=5, column=0, sticky='nsew')

    slider_frame.grid_rowconfigure(0, weight=1)
    slider_frame.grid_rowconfigure(1, weight=1)
    slider_frame.grid_rowconfigure(2, weight=1)
    slider_frame.grid_rowconfigure(3, weight=1)
    slider_frame.grid_rowconfigure(4, weight=1)
    slider_frame.grid_rowconfigure(5, weight=1)
    slider_frame.grid_columnconfigure(0, weight=1)

    # Inizia a mostrare il video trasformato
    show_transformed_video(video_paths.get())


# Funzioni per normalizzare angoli theta (-180,180) e phi (-90, 90) e fov (5, 90)
def normalize_theta(theta):
    # Normalizza theta nel range [-180, 180]
    while theta > 180:
        theta -= 360
    while theta < -180:
        theta += 360
    return theta

def normalize_phi(phi):
    #Limita l'ampiezza di phi per evitare l'inversione dell'immagine
    return max(-90, min(90, phi))

def normalize_fov(fov):
    return max(5, min(90, fov))


# Variabili globali per gestire lo stato della pressione dei tasti
left_pressed = False
right_pressed = False
up_pressed = False
down_pressed = False
zoom_in_pressed = False
zoom_out_pressed = False

def on_key_press(event):
    global left_pressed, right_pressed, up_pressed, down_pressed, zoom_in_pressed, zoom_out_pressed
    if event.keysym == 'Left':
        left_pressed = True
        update_theta_continuously(-1) # Aggiorna theta incrementandolo
    elif event.keysym == 'Right':
        right_pressed = True
        update_theta_continuously(1) # Aggiorna theta decrementandolo
    elif event.keysym == 'Up':
        up_pressed = True
        update_phi_continuously(1)  # Aggiorna phi incrementandolo
    elif event.keysym == 'Down':
        down_pressed = True
        update_phi_continuously(-1)  # Aggiorna phi decrementandolo
    elif event.char == '+':
        zoom_in_pressed = True
        update_fov_continuously(-1)  # Decrementa fov per zoomare
    elif event.char == '-':
        zoom_out_pressed = True
        update_fov_continuously(1)   # Incrementa fov per zoom out

def on_key_release(event):
    global left_pressed, right_pressed, up_pressed, down_pressed, zoom_in_pressed, zoom_out_pressed
    if event.keysym == 'Left':
        left_pressed = False
    elif event.keysym == 'Right':
        right_pressed = False
    elif event.keysym == 'Up':
        up_pressed = False
    elif event.keysym == 'Down':
        down_pressed = False
    elif event.char == '+':
        zoom_in_pressed = False
    elif event.char == '-':
        zoom_out_pressed = False

def update_phi_continuously(direction):
    global phi
    if (up_pressed and direction == 1) or (down_pressed and direction == -1):
        phi += direction
        phi = normalize_phi(phi)  
        phi_slider.set(phi)

def update_fov_continuously(amount):
    global fov
    if (zoom_in_pressed and amount < 0) or (zoom_out_pressed and amount > 0):
        fov += amount
        fov = normalize_fov(fov)  # Limita fov tra 5 e 90
        fov_slider.set(fov)

def update_theta_continuously(direction):
    global theta
    if (left_pressed and direction == -1) or (right_pressed and direction == 1):
        theta += direction
        theta = normalize_theta(theta)
        # Aggiorna lo slider di theta e l'immagine
        theta_slider.set(theta)
        

#Per l'aggiornamento dell'immagine dall'input testuale senza premere il button
def on_enter(event):
    update_with_input()


# Modifica la chiamata iniziale del programma
if __name__ == "__main__":
    setup_main_window()