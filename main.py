import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Reshape
from tensorflow.keras.utils import to_categorical
import tkinter as tk
from tkinter import Canvas, Button, Label, Frame, Scale, HORIZONTAL
from PIL import Image, ImageDraw, ImageGrab, ImageFilter
import io
import os
import random
from tkinter.simpledialog import askinteger
from tkinter import filedialog

# Définition des données d'entraînement
# Codage des chiffres et lettres en matrice 7x5
def create_digit_data():
    # Définition des chiffres de 0 à 9 en matrices 7x5
    digits = {
        0: [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ],
        1: [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1]
        ],
        2: [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1]
        ],
        3: [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ],
        4: [
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1]
        ],
        5: [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ],
        6: [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ],
        7: [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ],
        8: [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ],
        9: [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ],
        10: [  # A
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1]
        ],
        11: [  # B
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 0]
        ],
        12: [  # C
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        13: [  # D
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 0]
        ],
        14: [  # E
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1]
        ],
        15: [  # F
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0]
        ],
        16: [  # G
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        17: [  # H
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1]
        ],
        18: [  # I
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1]
        ],
        19: [  # J
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0]
        ],
        20: [  # K
            [1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1]
        ],
        21: [  # L
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1]
        ],
        22: [  # M
            [1, 0, 0, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1]
        ],
        23: [  # N
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1]
        ],
        24: [  # O
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        25: [  # P
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0]
        ],
        26: [  # Q
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 1]
        ],
        27: [  # R
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1]
        ],
        28: [  # S
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        29: [  # T
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ],
        30: [  # U
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        31: [  # V
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0]
        ],
        32: [  # W
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1]
        ],
        33: [  # X
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1]
        ],
        34: [  # Y
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ],
        35: [  # Z
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1]
        ]
    }
    
    # Convertir en tableaux numpy
    X = []
    y = []
    
    for digit, pattern in digits.items():
        X.append(pattern)
        y.append(digit)
    
    return np.array(X), np.array(y)

# Fonction pour ajouter du bruit aux données
def add_noise(data, noise_factor=0.2):
    noisy_data = data.copy()
    for i in range(len(noisy_data)):
        # Ajouter du bruit aléatoire
        noise = np.random.rand(*noisy_data[i].shape) < noise_factor
        # Inverser les bits où il y a du bruit
        noisy_data[i] = np.logical_xor(noisy_data[i], noise).astype(int)
    return noisy_data

# Fonction pour générer des variations des données d'origine
def generate_variations(X, y, num_variations=5, noise_range=(0.05, 0.2)):
    X_variations = []
    y_variations = []
    
    for i in range(len(X)):
        # Ajouter l'original
        X_variations.append(X[i])
        y_variations.append(y[i])
        
        # Ajouter des variations avec différents niveaux de bruit
        for j in range(num_variations):
            noise_level = random.uniform(noise_range[0], noise_range[1])
            noisy_sample = add_noise(np.array([X[i]]), noise_level)[0]
            X_variations.append(noisy_sample)
            y_variations.append(y[i])
            
    return np.array(X_variations), np.array(y_variations)

# Création du modèle optimisé
def create_model():
    model = Sequential([
        Flatten(input_shape=(7, 5)),  # Aplatir l'entrée 7x5
        Dense(128, activation='relu'),  # Augmenter la taille des couches cachées
        Dropout(0.3),  # Ajouter du dropout pour éviter le surapprentissage
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(36, activation='softmax')  # 36 classes: 0-9 et A-Z
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Interface graphique pour l'apprentissage et le test
class DigitRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconnaissance de Chiffres")
        self.root.geometry("800x700")
        
        # Créer le modèle
        self.model = create_model()
        
        # Créer les données de base
        self.X_train_base, self.y_train_base = create_digit_data()
        
        # Générer des variations pour l'entraînement
        self.X_train, self.y_train = generate_variations(self.X_train_base, self.y_train_base, num_variations=10)
        
        # Définir le mode de dessin par défaut
        self.drawing_mode = tk.StringVar(value="libre")
        
        # Initialiser les variables pour le mode pixel
        self.pixel_matrix = np.zeros((7, 5), dtype=np.int32)
        self.pixel_rectangles = []
        
        # Interface
        self.create_widgets()
        
        # Variables pour le dessin
        self.old_x = None
        self.old_y = None
        self.line_width = 15
        self.color = "black"
        
        # Historique des prédictions
        self.predictions_history = []
        
        # Initialiser la grille de dessin en mode libre par défaut
        self.root.update()  # Forcer la mise à jour pour que les dimensions soient correctes
        self.draw_guide_grid()
    
    def create_widgets(self):
        # Frame principale
        main_frame = Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Frame gauche pour le dessin et les contrôles
        left_frame = Frame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=10)
        
        # Frame droite pour les résultats et visualisations
        right_frame = Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=10)
        
        # Titre
        title_label = Label(left_frame, text="Reconnaissance de Chiffres", font=("Arial", 18, "bold"))
        title_label.pack(pady=10)
        
        # Canvas pour le dessin
        self.canvas_frame = Frame(left_frame, bd=2, relief="groove")
        self.canvas_frame.pack(padx=10, pady=10)
        
        self.canvas = Canvas(self.canvas_frame, width=280, height=280, bg="white", bd=3, relief="ridge")
        self.canvas.pack()
        
        # Gestionnaires d'événements pour le dessin
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # Ajouter une grille de guidage
        self.draw_guide_grid()
        
        # Contrôle de l'épaisseur du trait
        width_frame = Frame(left_frame)
        width_frame.pack(pady=5)
        
        width_label = Label(width_frame, text="Épaisseur du trait:")
        width_label.pack(side="left", padx=5)
        
        self.width_scale = Scale(width_frame, from_=1, to=30, orient=HORIZONTAL, length=200, command=self.change_width)
        self.width_scale.set(15)
        self.width_scale.pack(side="left")
        
        # Boutons de contrôle
        control_frame = Frame(left_frame)
        control_frame.pack(pady=10)
        
        clear_button = Button(control_frame, text="Effacer", command=self.clear_canvas, width=10, bg="#ff9999")
        clear_button.grid(row=0, column=0, padx=5)
        
        recognize_button = Button(control_frame, text="Reconnaître", command=self.recognize_drawing, width=10, bg="#99ccff")
        recognize_button.grid(row=0, column=1, padx=5)
        
        adjust_button = Button(control_frame, text="Ajuster", command=self.adjust_drawing, width=10, bg="#ffcc99")
        adjust_button.grid(row=0, column=2, padx=5)
        
        save_example_button = Button(control_frame, text="Enregistrer exemple", 
                                   command=self.save_character_dialog, width=15, bg="#99ccff")
        save_example_button.grid(row=1, column=0, padx=5, pady=5)
        
        # Frame pour les boutons d'apprentissage
        train_frame = Frame(left_frame)
        train_frame.pack(pady=10)
        
        train_button = Button(train_frame, text="Entraîner le réseau", command=self.train_model, width=15, bg="#99ff99")
        train_button.grid(row=0, column=0, padx=5)
        
        test_button = Button(train_frame, text="Tester avec bruit", command=self.test_model, width=15, bg="#ffcc99")
        test_button.grid(row=0, column=1, padx=5)
        
        # Mode de dessin (radio buttons)
        mode_frame = Frame(left_frame)
        mode_frame.pack(pady=5)
        
        mode_label = Label(mode_frame, text="Mode de dessin:")
        mode_label.pack(side="left", padx=5)
        
        free_mode = tk.Radiobutton(mode_frame, text="Libre", variable=self.drawing_mode, 
                                  value="libre", command=self.change_drawing_mode)
        free_mode.pack(side="left")
        
        pixel_mode = tk.Radiobutton(mode_frame, text="Pixel par pixel", variable=self.drawing_mode, 
                                   value="pixel", command=self.change_drawing_mode)
        pixel_mode.pack(side="left")
        
        # Ajouter après la ligne 506 (après la création de train_frame)
        library_button = Button(train_frame, text="Charger Bibliothèque", 
                               command=self.load_image_library, width=15, bg="#cc99ff")
        library_button.grid(row=0, column=2, padx=5)
        
        # Label pour afficher le résultat
        self.result_frame = Frame(right_frame, bd=2, relief="groove", padx=10, pady=10)
        self.result_frame.pack(fill="x", pady=10)
        
        self.result_title = Label(self.result_frame, text="Résultat de la prédiction", font=("Arial", 14, "bold"))
        self.result_title.pack()
        
        self.result_label = Label(self.result_frame, text="En attente...", font=("Arial", 36))
        self.result_label.pack(pady=5)
        
        self.confidence_label = Label(self.result_frame, text="")
        self.confidence_label.pack()
        
        # Frame pour visualiser la matrice
        self.matrix_frame = Frame(right_frame, bd=2, relief="groove", padx=10, pady=10)
        self.matrix_frame.pack(fill="x", pady=10)
        
        self.matrix_title = Label(self.matrix_frame, text="Matrice 7x5 extraite", font=("Arial", 14, "bold"))
        self.matrix_title.pack()
        
        self.matrix_canvas = Canvas(self.matrix_frame, width=150, height=210, bg="white")
        self.matrix_canvas.pack(pady=10)
        
        # Frame pour l'historique des prédictions
        self.history_frame = Frame(right_frame, bd=2, relief="groove", padx=10, pady=10)
        self.history_frame.pack(fill="both", expand=True, pady=10)
        
        self.history_title = Label(self.history_frame, text="Historique des prédictions", font=("Arial", 14, "bold"))
        self.history_title.pack()
        
        self.history_label = Label(self.history_frame, text="Aucune prédiction", justify="left")
        self.history_label.pack(pady=5, anchor="w")
    
    def start_draw(self, event):
        """Commence le dessin"""
        # Ne rien faire en mode pixel (les pixels ont leurs propres gestionnaires)
        if self.drawing_mode.get() == "pixel":
            return
        
        self.old_x = event.x
        self.old_y = event.y
    
    def draw(self, event):
        """Continue le dessin"""
        # Ne rien faire en mode pixel
        if self.drawing_mode.get() == "pixel":
            return
        
        if self.old_x and self.old_y:
            self.canvas.create_line(
                self.old_x, self.old_y, event.x, event.y,
                width=self.line_width, fill=self.color,
                capstyle="round", smooth=True
            )
        
        self.old_x = event.x
        self.old_y = event.y
        
        # Mettre à jour l'aperçu de la matrice toutes les 5 actions de dessin
        self.draw_counter = getattr(self, 'draw_counter', 0) + 1
        if self.draw_counter % 5 == 0:
            self.update_matrix_preview()
    
    def stop_draw(self, event):
        """Termine le dessin"""
        # Ne rien faire en mode pixel
        if self.drawing_mode.get() == "pixel":
            return
        
        self.old_x = None
        self.old_y = None
    
    def change_width(self, val):
        self.line_width = int(val)
    
    def clear_canvas(self):
        if hasattr(self, 'drawing_mode') and self.drawing_mode.get() == "pixel":
            # En mode pixel, réinitialiser la matrice mais garder la grille
            if hasattr(self, 'pixel_matrix') and hasattr(self, 'pixel_rectangles'):
                for i in range(7):
                    for j in range(5):
                        if i < len(self.pixel_matrix) and j < len(self.pixel_matrix[i]):
                            self.pixel_matrix[i, j] = 0
                            if i < len(self.pixel_rectangles) and j < len(self.pixel_rectangles[i]):
                                rect_id = self.pixel_rectangles[i][j]
                                self.canvas.itemconfig(rect_id, fill="white")
        else:
            # En mode libre, tout effacer sauf la grille
            self.canvas.delete("all")
            self.draw_guide_grid()
        
        self.old_x = None
        self.old_y = None
        self.result_label.config(text="En attente...")
        self.confidence_label.config(text="")
        self.matrix_canvas.delete("all")
    
    def train_model(self):
        # Entraîner le modèle avec des callbacks pour suivre la progression
        self.result_label.config(text="Entraînement...")
        self.root.update()
        
        # Augmenter les données d'entraînement avec des variations
        X_augmented, y_augmented = generate_variations(self.X_train_base, self.y_train_base, num_variations=5)
        
        # Réinitialiser les poids du modèle pour éviter les biais d'apprentissage précédents
        self.model = create_model()
        
        # Entraîner le modèle avec plus d'époques et un taux d'apprentissage plus faible
        optimizer = keras.optimizers.Adam(learning_rate=0.0005)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Utiliser une validation croisée pour éviter le surapprentissage
        history = self.model.fit(
            X_augmented, y_augmented,
            epochs=100,  # Plus d'époques
            batch_size=16,  # Batch size plus petit
            validation_split=0.2,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        
        # Afficher les résultats
        val_acc = history.history['val_accuracy'][-1]
        self.result_label.config(text=f"Entraîné!")
        self.confidence_label.config(text=f"Précision de validation: {val_acc:.2f}")
        
        # Sauvegarder le modèle
        self.model.save("digit_recognition_model.h5")
    
    def test_model(self):
        # Tester avec des données bruitées
        X_test = add_noise(self.X_train_base, 0.2)
        
        # Prédictions
        y_pred = self.model.predict(X_test)
        predictions = np.argmax(y_pred, axis=1)
        
        # Calculer la précision
        accuracy = np.mean(predictions == self.y_train_base)
        
        # Afficher les résultats
        self.result_label.config(text=f"Test")
        self.confidence_label.config(text=f"Précision avec bruit: {accuracy:.2f}")
        
        # Visualiser un exemple bruité
        random_idx = np.random.randint(0, len(X_test))
        self.visualize_matrix(X_test[random_idx])
        
        # Ajouter au historique
        self.add_to_history(f"Test avec bruit: {accuracy:.2f}")
    
    def recognize_drawing(self):
        try:
            # Modifier pour utiliser la matrice de pixels en mode pixel
            if hasattr(self, 'drawing_mode') and self.drawing_mode.get() == "pixel" and hasattr(self, 'pixel_matrix'):
                matrix = self.pixel_matrix.copy()
            else:
                matrix = self.canvas_to_digit()
            
            if matrix is None:
                self.result_label.config(text="Erreur")
                self.confidence_label.config(text="Impossible de capturer le dessin")
                return
            
            # Visualiser la matrice extraite
            self.visualize_matrix(matrix)
            
            # Prédire
            prediction = self.model.predict(np.array([matrix]), verbose=0)
            class_idx = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            # Convertir l'indice en caractère (0-9, A-Z)
            if class_idx < 10:
                result_char = str(class_idx)  # Chiffres 0-9
            else:
                result_char = chr(class_idx - 10 + ord('A'))  # Lettres A-Z
            
            # Afficher le résultat
            self.result_label.config(text=f"{result_char}")
            self.confidence_label.config(text=f"Confiance: {confidence:.2f}")
            
            # Afficher les 3 meilleures prédictions
            top3_indices = np.argsort(prediction[0])[-3:][::-1]
            top3_values = prediction[0][top3_indices]
            
            # Convertir les indices en caractères
            top3_chars = []
            for idx in top3_indices:
                if idx < 10:
                    top3_chars.append(str(idx))
                else:
                    top3_chars.append(chr(idx - 10 + ord('A')))
            
            top3_text = ", ".join([f"{char}({val:.2f})" for char, val in zip(top3_chars, top3_values)])
            
            # Ajouter au historique
            self.add_to_history(f"Prédiction: {result_char} (Top3: {top3_text})")
        except Exception as e:
            print(f"Erreur lors de la reconnaissance: {e}")
            self.result_label.config(text="Erreur")
            self.confidence_label.config(text=str(e))
    
    def canvas_to_digit(self):
        try:
            # Obtenir les dimensions du canvas
            x = self.canvas.winfo_rootx() + self.canvas.winfo_x()
            y = self.canvas.winfo_rooty() + self.canvas.winfo_y()
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()
            
            # Capturer l'image du canvas
            image = ImageGrab.grab(bbox=(x, y, x+width, y+height))
            
            # Convertir en niveaux de gris
            image = image.convert('L')
            
            # Appliquer un flou pour réduire le bruit
            image = image.filter(ImageFilter.GaussianBlur(radius=1))
            
            # Redimensionner directement en 7x5 pour correspondre à notre grille
            image = image.resize((5, 7), Image.LANCZOS)
            
            # Convertir en tableau numpy
            img_array = np.array(image)
            
            # Inverser les couleurs (fond blanc, chiffre noir -> fond noir, chiffre blanc)
            img_array = 255 - img_array
            
            # Normaliser et binariser avec un seuil adaptatif
            threshold = np.mean(img_array) * 0.5  # Seuil adaptatif
            matrix = (img_array > threshold).astype(np.int32)
            
            return matrix
            
        except Exception as e:
            print(f"Erreur lors de la capture du dessin: {e}")
            return None
    
    def visualize_matrix(self, matrix):
        # Effacer le canvas
        self.matrix_canvas.delete("all")
        
        # Taille des cellules
        cell_width = 20
        cell_height = 20
        padding = 5
        
        # Dessiner la matrice avec des couleurs plus contrastées
        for i in range(7):
            for j in range(5):
                x0 = j * cell_width + padding
                y0 = i * cell_height + padding
                x1 = x0 + cell_width
                y1 = y0 + cell_height
                
                # Utiliser un dégradé de couleurs pour montrer l'intensité
                if isinstance(matrix[i, j], (int, np.integer)):
                    # Valeur binaire
                    color = "black" if matrix[i, j] == 1 else "white"
                    self.matrix_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="gray")
                else:
                    # Valeur continue (probabilité)
                    intensity = int(255 * (1 - matrix[i, j]))
                    color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
                    self.matrix_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="gray")
    
    def add_to_history(self, text):
        # Ajouter une prédiction à l'historique
        self.predictions_history.append(text)
        
        # Limiter l'historique aux 5 dernières prédictions
        if len(self.predictions_history) > 5:
            self.predictions_history = self.predictions_history[-5:]
        
        # Mettre à jour l'affichage
        history_text = "\n".join(self.predictions_history)
        self.history_label.config(text=history_text)

    def draw_guide_grid(self):
        # Dessiner une grille 7x5 légère pour guider le dessin
        self.canvas.delete("grid")  # Supprimer l'ancienne grille
        
        width = self.canvas.winfo_width() or 280
        height = self.canvas.winfo_height() or 280
        
        cell_width = width / 5
        cell_height = height / 7
        
        # Lignes horizontales
        for i in range(8):
            y = i * cell_height
            self.canvas.create_line(0, y, width, y, fill="#DDDDDD", dash=(2, 4), tags="grid")
        
        # Lignes verticales
        for i in range(6):
            x = i * cell_width
            self.canvas.create_line(x, 0, x, height, fill="#DDDDDD", dash=(2, 4), tags="grid")

    def change_drawing_mode(self):
        try:
            self.clear_canvas()
            mode = self.drawing_mode.get()
            
            if mode == "pixel":
                # Créer une grille de pixels 7x5
                self.create_pixel_grid()
            else:
                # Revenir au mode de dessin libre
                self.draw_guide_grid()
        except Exception as e:
            print(f"Erreur lors du changement de mode: {e}")
            # Réinitialiser en mode libre en cas d'erreur
            self.drawing_mode.set("libre")
            self.canvas.delete("all")
            self.draw_guide_grid()

    def create_pixel_grid(self):
        self.canvas.delete("all")
        
        width = self.canvas.winfo_width() or 280
        height = self.canvas.winfo_height() or 280
        
        cell_width = width / 5
        cell_height = height / 7
        
        # Créer une matrice 7x5 de rectangles cliquables
        self.pixel_matrix = np.zeros((7, 5), dtype=np.int32)
        self.pixel_rectangles = []
        
        for i in range(7):
            row_rects = []
            for j in range(5):
                x0 = j * cell_width + 2
                y0 = i * cell_height + 2
                x1 = (j + 1) * cell_width - 2
                y1 = (i + 1) * cell_height - 2
                
                rect_id = self.canvas.create_rectangle(
                    x0, y0, x1, y1, 
                    fill="white", outline="gray",
                    tags=f"pixel_{i}_{j}"
                )
                row_rects.append(rect_id)
                
                # Utiliser tag_bind au lieu de bind pour éviter les conflits
                self.canvas.tag_bind(
                    f"pixel_{i}_{j}", 
                    "<Button-1>", 
                    lambda event, i=i, j=j: self.toggle_pixel(i, j)
                )
            
            self.pixel_rectangles.append(row_rects)

    def toggle_pixel(self, i, j):
        """Inverse l'état d'un pixel"""
        try:
            # Inverser l'état du pixel
            self.pixel_matrix[i, j] = 1 - self.pixel_matrix[i, j]
            
            # Mettre à jour l'affichage
            rect_id = self.pixel_rectangles[i][j]
            color = "black" if self.pixel_matrix[i, j] == 1 else "white"
            self.canvas.itemconfig(rect_id, fill=color)
            
            # Mettre à jour l'aperçu de la matrice
            self.visualize_matrix(self.pixel_matrix)
        except Exception as e:
            print(f"Erreur lors du toggle du pixel ({i},{j}): {e}")

    def update_matrix_preview(self):
        # Capturer la matrice actuelle et la visualiser
        matrix = self.canvas_to_digit()
        if matrix is not None:
            self.visualize_matrix(matrix)

    def adjust_drawing(self):
        """Ajuste le dessin pour mieux correspondre aux modèles de référence"""
        if self.drawing_mode.get() == "pixel":
            matrix = self.pixel_matrix.copy()
        else:
            matrix = self.canvas_to_digit()
        
        if matrix is None:
            return
        
        # Prédire avec le modèle actuel
        prediction = self.model.predict(np.array([matrix]))

    def save_character_dialog(self):
        """Ouvre une boîte de dialogue pour enregistrer un caractère"""
        # Créer une fenêtre de dialogue personnalisée
        dialog = tk.Toplevel(self.root)
        dialog.title("Enregistrer exemple")
        dialog.geometry("300x200")
        dialog.resizable(False, False)
        
        # Ajouter des instructions
        label = Label(dialog, text="Quel caractère avez-vous dessiné?")
        label.pack(pady=10)
        
        # Créer des boutons pour les chiffres
        digits_frame = Frame(dialog)
        digits_frame.pack(pady=5)
        
        for i in range(10):
            btn = Button(digits_frame, text=str(i), width=2,
                        command=lambda i=i: [dialog.destroy(), self.save_character(i)])
            btn.grid(row=0, column=i, padx=2)
        
        # Créer des boutons pour les lettres
        letters_frame = Frame(dialog)
        letters_frame.pack(pady=5)
        
        for i in range(26):
            letter = chr(ord('A') + i)
            btn = Button(letters_frame, text=letter, width=2,
                        command=lambda idx=i+10: [dialog.destroy(), self.save_character(idx)])
            btn.grid(row=0 if i < 13 else 1, column=i % 13, padx=2)
        
        # Bouton d'annulation
        cancel_btn = Button(dialog, text="Annuler", command=dialog.destroy)
        cancel_btn.pack(pady=10)
        
        # Centrer la fenêtre
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        
        # Rendre la fenêtre modale
        dialog.transient(self.root)
        dialog.grab_set()
        self.root.wait_window(dialog)

    def save_character(self, char_idx):
        """Enregistre le caractère avec l'indice donné"""
        # Récupérer la matrice
        if self.drawing_mode.get() == "pixel":
            matrix = self.pixel_matrix.copy()
        else:
            matrix = self.canvas_to_digit()
        
        if matrix is None:
            return
        
        # Ajouter aux données d'entraînement avec un poids plus important
        # Ajouter plusieurs copies pour donner plus d'importance à cet exemple
        char_count = 0
        for _ in range(5):  # Ajouter 5 copies
            self.X_train_base = np.append(self.X_train_base, [matrix], axis=0)
            self.y_train_base = np.append(self.y_train_base, [char_idx], axis=0)
        
        # Convertir l'indice en caractère
        if char_idx < 10:
            char = str(char_idx)
        else:
            char = chr(char_idx - 10 + ord('A'))
        
        # Informer l'utilisateur
        self.add_to_history(f"Exemple de '{char}' enregistré (x5)")
        
        # Option: réentraîner immédiatement
        if tk.messagebox.askyesno("Réentraîner", "Voulez-vous réentraîner le modèle maintenant?"):
            self.train_model()

    def load_image_library(self):
        """Charge une bibliothèque d'images organisée en dossiers par caractère"""
        # Sélectionner le dossier principal de la bibliothèque
        library_path = filedialog.askdirectory(title="Sélectionner le dossier de la bibliothèque d'images")
        
        if not library_path:
            return
        
        try:
            # Compteurs pour le suivi
            total_added = 0
            chars_added = []
            
            # Parcourir les sous-dossiers (un par caractère)
            for char_folder in os.listdir(library_path):
                folder_path = os.path.join(library_path, char_folder)
                
                # Vérifier que c'est un dossier et que le nom est valide
                if not os.path.isdir(folder_path):
                    continue
                    
                # Déterminer l'index du caractère
                char_idx = None
                if char_folder.isdigit() and len(char_folder) == 1:
                    # C'est un chiffre de 0 à 9
                    char_idx = int(char_folder)
                elif len(char_folder) == 1 and 'A' <= char_folder.upper() <= 'Z':
                    # C'est une lettre de A à Z
                    char_idx = ord(char_folder.upper()) - ord('A') + 10
                
                if char_idx is None or char_idx >= 36:
                    continue  # Caractère non pris en charge
                    
                # Compteur d'images pour ce caractère
                char_count = 0
                
                # Parcourir les images dans ce dossier
                for img_file in os.listdir(folder_path):
                    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        continue
                        
                    img_path = os.path.join(folder_path, img_file)
                    
                    try:
                        # Ouvrir et prétraiter l'image
                        img = Image.open(img_path).convert('L')  # Niveaux de gris
                        img = img.resize((5, 7), Image.LANCZOS)  # Redimensionner à 5x7
                        
                        # Convertir en tableau numpy et binariser
                        img_array = np.array(img)
                        # Inverser si nécessaire (fond blanc, caractère noir)
                        if np.mean(img_array) > 128:
                            img_array = 255 - img_array
                        
                        # Binariser avec seuil adaptatif
                        threshold = np.mean(img_array) * 0.7
                        matrix = (img_array > threshold).astype(np.int32)
                        
                        # Ajouter aux données d'entraînement
                        self.X_train_base = np.append(self.X_train_base, [matrix], axis=0)
                        self.y_train_base = np.append(self.y_train_base, [char_idx], axis=0)
                        
                        char_count += 1
                        total_added += 1
                        
                    except Exception as e:
                        print(f"Erreur lors du traitement de {img_path}: {e}")
                
                if char_count > 0:
                    chars_added.append(f"{char_folder}({char_count})")
            
            # Rapport de chargement
            if total_added > 0:
                chars_text = ", ".join(chars_added)
                self.add_to_history(f"Bibliothèque chargée: {total_added} images ({chars_text})")
                
                # Proposer l'entraînement
                if tk.messagebox.askyesno("Entraînement", 
                    f"{total_added} images ont été ajoutées. Voulez-vous entraîner le modèle maintenant?"):
                    self.train_model()
            else:
                tk.messagebox.showinfo("Information", 
                    "Aucune image valide n'a été trouvée dans la bibliothèque.")
                
        except Exception as e:
            tk.messagebox.showerror("Erreur", 
                f"Erreur lors du chargement de la bibliothèque: {str(e)}")

# Exécution de l'application
if __name__ == "__main__":
    # Vérifier si un modèle sauvegardé existe
    if os.path.exists("digit_recognition_model.h5"):
        print("Modèle existant trouvé. Chargement...")
    else:
        print("Aucun modèle existant. Un nouveau modèle sera créé.")
    
    # Créer l'application
    root = tk.Tk()
    app = DigitRecognitionApp(root)
    root.mainloop()
