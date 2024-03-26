import cv2
import numpy as np
#Libreria per gestione spostamento e zoom
import keyboard
#Librerie per creazione e personalizzazione finestre
import tkinter as tk
from tkinter import ttk, PhotoImage



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
    x_out, y_out = np.meshgrid(np.arange(width_out), np.arange(height_out))

    # Normalizza le coordinate (da 0 a 1) e applica il FOV come proiettato sul piano tangente alla sfera
    x_out = ((x_out / width_out) - 0.5) * 2 * np.tan(fov_rad / 2)
    y_out = -((y_out / height_out) - 0.5) * 2 * np.tan(fov_rad / 2)

    # Calcola il punto P sul piano tangente canonico (z=1)
    P = np.stack((x_out, y_out, np.ones_like(x_out)), axis=-1)

    # Normalizza P per proiettarlo sulla sfera unitaria
    P /= np.linalg.norm(P, axis=2, keepdims=True)

    # Matrice di rotazione per theta (rotazione attorno all'asse y)
    R_theta = np.array([
        [np.cos(theta_rad), 0, -np.sin(theta_rad)],
        [0, 1, 0],
        [np.sin(theta_rad), 0, np.cos(theta_rad)]
    ])

    # Matrice di rotazione per phi (rotazione attorno all'asse x)
    R_phi = np.array([
        [1, 0, 0],
        [0, np.cos(phi_rad), -np.sin(phi_rad)],
        [0, np.sin(phi_rad), np.cos(phi_rad)]
    ])

    # Crea la matrice di rotazione combinata, prima phi e poi theta facendo la moltiplicazione tra matrici
    R = R_phi @ R_theta
    #Applico la rotazione combinata a ogni punto P
    #Moltiplicazione tra la matrice multidimensionale P e la matrice di rotazione R
    #ijk dimensione di P, kl dimensione di R, ijl dimensione di P_rotated. (ij da P, l da R)
    P_rotated = np.einsum('ijk,kl->ijl', P, R)

    # Calcola le coordinate equirettangolari equivalenti
    x_eq = w_ratio * (np.arctan2(P_rotated[..., 2], P_rotated[..., 0]) + np.pi/2)
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
    perspective_image = equirectangular_to_perspective(equirectangular, fov, theta, phi, height_out, width_out)
    if perspective_image is not None:
        show_image_on_label(perspective_image)
    else:
        print("Errore nell'aggiornamento dell'immagine.")

def show_image_on_label(img):
    cv2.imwrite('temp_image.png', img)
    tk_image = PhotoImage(file='temp_image.png')
    image_label.image = tk_image
    image_label['image'] = image_label.image

def update_with_input(*args):
    global theta, phi, fov, equirectangular, video_paths, theta_entry, phi_entry, fov_entry
    video_path = video_paths.get()  # Ottiene il percorso del video selezionato
    # Aggiorna le variabili globali con i valori inseriti dall'utente
    theta = int(theta_entry.get())
    phi = int(phi_entry.get())
    fov = int(fov_entry.get())
    # Estrae nuovamente il frame equirettangolare usando il percorso del video aggiornato
    equirectangular = extract_frame_equirectangular(video_path, frame_number)
    update_image()
    update_status_label()

def update_status_label():
    # Aggiorna il testo del label con i valori correnti di theta, phi e fov
    status_text = f"Theta: {int(theta)}°, Phi: {int(phi)}°, FOV: {int(fov)}°"
    status_label.config(text=status_text)


# Setup iniziale
video_path = 'video_1.MP4'
frame_number = 0 #un frame randome prelevato dal video
# Angoli di rotazione in gradi (theta attorno all'asse verticale (longitudine), Phi l'inclinazione su e giù (latitudine))
theta = 0
phi = 0
fov = 70  # Campo visivo in gradi (field of view)
height_out = 600  # Altezza dell'immagine di output
width_out = 1200   # Larghezza dell'immagine di output


# Creazione e impostazione della finestra con libreria tkinter
window = tk.Tk()
window.title("Visualizzatore immagine prospettica")
window.geometry('1250x800+0+0')  # 1200x800 sono le dimensioni, +0+0 posiziona la finestra in alto a sinistra


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
- Usa le frecce destra/sinistra per spostarti orizzontalmente di 10° (theta).
- Usa le frecce su/giù per spostarti verticalmente di 5° (phi).
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


# Funzioni per normalizzare angoli theta (-180,180) e phi (-90, 90)
def normalize_theta(theta):
    if theta > 350:
        theta = 0
    elif theta < -350:
        theta = 0
    return theta

def normalize_phi(phi):
    #Limita l'ampiezza di phi per evitare l'inversione dell'immagine
    if phi > 90:
        phi = 90
    elif phi < -90:
        phi = -90
    return phi

# Funzioni per lo spostamento e lo zoom dell'immagine prospettica ricavata
def on_right_arrow(event):
    global theta
    theta += 10
    theta = normalize_theta(theta)
    update_image()
    update_status_label()  # Aggiorna il label dello stato

def on_left_arrow(event):
    global theta
    theta -= 10
    theta = normalize_theta(theta)
    update_image()
    update_status_label()


def on_up_arrow(event):
    global phi
    phi += 5
    phi = normalize_phi(phi)
    update_image()
    update_status_label()

def on_down_arrow(event):
    global phi
    phi -= 5
    phi = normalize_phi(phi)
    update_image()
    update_status_label()


def on_zoom_in(event):
    global fov
    if fov > 10:  # Imposta un limite inferiore per evitare uno zoom in eccessivo
        fov -= 5  
    update_image()
    update_status_label()

def on_zoom_out(event):
    global fov
    if fov < 110:  #Zoom out fino a 110° per evitare distorsioni
        fov += 5  
    update_image()
    update_status_label()

#Per l'aggiornamento dell'immagine senza premere il button
def on_enter(event):
    update_with_input()


# Ascolta le pressioni dei tasti
keyboard.on_press_key("right", on_right_arrow)
keyboard.on_press_key("left", on_left_arrow)
keyboard.on_press_key("up", on_up_arrow)
keyboard.on_press_key("down", on_down_arrow)
keyboard.on_press_key("+", on_zoom_in)
keyboard.on_press_key("-", on_zoom_out)
keyboard.on_press_key("enter", on_enter)



# Estrai il frame equirettangolare
equirectangular = extract_frame_equirectangular(video_paths.get(), frame_number)
update_image() #Per mostrare l'immagine iniziale predefinita dal setup iniziale
update_status_label() # Per mostrare i valori iniziali di theta, phi e fov

# avvia il loop degli eventi di Tkinter per ascoltare modifiche e interazioni dell'utente
window.mainloop()