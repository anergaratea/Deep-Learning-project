# 🧬 Molecular Graph Neural Network (Molecular GNN)

Este repositorio implementa una Graph Neural Network (GNN) avanzada para predecir propiedades químicas de moléculas a partir de su representación en formato SMILES. Se basa en el aprendizaje de representaciones estructurales mediante grafos extraídos con **RDKit** y procesados con **PyTorch**.

## ✨ Características Principales
- **Preprocesamiento Químico**: Conversión automática de strings SMILES a grafos (matriz de características de átomos y enlaces) utilizando `RDKit`.
- **Arquitecturas Deep Learning**: Implementación desde cero de *Message Passing Neural Networks* convolucionales (`MolecularGNN`) y *Graph Attention Networks* (`MolecularGAT`).
- **Framework PyTorch**: Configuración completa con un DataLoader a medida (`collate_fn`) para poder procesar "batches" de grafos de distintos tamaños eficientemente.
- **Comparativa de Rendimiento (Baseline)**: Incluye regresores de *Machine Learning* clásico (*Random Forest*) combinados con *Morgan Fingerprints* (huellas moleculares) mediante `scikit-learn`.
- **Integración con Datos Reales**: Descarga en tiempo de ejecución del dataset *FreeSolv (SAMPL)* (~642 compuestos con energía libre experimental de hidratación).
- **Visualizaciones Integradas**: Creación de curvas de pérdida (Loss), gráficos de concordancia "Valor Real vs. Predicho" (MAE/RMSE), además de dibujos 2D automáticos de la molécula interactuando con `matplotlib`.

## 🛠 Instalación

Una vez tengas clonado o descargado el repositorio, abre un terminal en el directorio raíz del proyecto y sigue estos pasos:

```bash
# 1. Crear entorno virtual
python -m venv venv

# 2. Activar entorno virtual
# En Windows (Powershell)
.\venv\Scripts\activate
# En Linux/Mac
source venv/bin/activate

# 3. Instalar las dependencias (PyTorch, RDKit, scikit-learn, etc.)
pip install -r requirements.txt
```

## 🚀 Ejemplos de Ejecución

Debes tener el entorno virtual activado previamente a ejecutar cualquiera de los scripts.

### 1. Entrenar el Modelo Baseline clásico
Para obtener una métrica base de error (MAE y RMSE) evaluando un *Random Forest* regressor sobre las huellas químicas tradicionales:
```bash
python baseline.py
```
> Retorna un modelo entrenado sobre características moleculares simples. El objetivo es comprobar si la red neuronal basada en grafos puede batir a este modelo o aprender estructuras más complejas.

### 2. Entrenar la Red Neuronal de Grafos (GNN)
El script principal evalúa nuestra red transformando grafos y cargando dinámicamente el set de datos real *FreeSolv*.

```bash
python train.py
```
**Comportamiento y Salidas:** 
El script procesará los strings de SMILES convirtiéndolos a nodos matemáticos, y entrenará la red GNN por 100 épocas. La pérdida (Loss) se imprimirá progresivamente en consola. 
Al finalizar, se guardarán 3 visualizaciones en la carpeta de tu proyecto de forma automática:
- `molecule.png`: Render interactivo de la primera molécula del fichero en plano 2D.
- `training_metrics.png`: Una métrica con la evolución del entreno comparando `Train Loss` frente a `Validation Loss`.
- `true_vs_predicted.png`: Gráfico que mapea las predicciones de la GNN frente a los valores reales esperados del dataset (para observar de manera explícita el ajuste y precisión de la predicción química).

### 3. Usar el Modelo Avanzado de Atención a Grafos (GAT)
Nuestra implementación extra incluye una red de atención gráfica avanzada (`MolecularGAT`) que pondera algorítmicamente y le da importancia distinta a los átomos vecinos según su contexto antes de agrupar al nivel molecular, buscando relaciones no-lineales subyacentes.

Para que `train.py` entrene esta topología, simplemente modifica la **sección de ajuste (aprox. línea 79)** en el archivo `train.py`:
```python
# Cambia esto en el archivo train.py
use_gat = True # Modifica 'False' por 'True' para habilitar GAT
```
Guarda el archivo y ejecuta nuevamente el pipeline de entrenamiento:
```bash
python train.py
```

## 📂 Organización del Proyecto

```text
Deep-Learning-project/
│
├── data/
│   ├── preprocessing.py  # Conversor de la cadena SMILES a Matriz Numpy/PyTorch Tensor.
│   └── dataset.py        # Descargador/Loader del CSV real para la predicción de hidratación química.
│
├── models/
│   ├── gnn.py            # Deep Neural Network por convención Message-Passing (Aglutinador GCN).
│   └── gat.py            # Grafo Neural Atencional especializado (Foco en vecinos importantes).
│
├── utils/
│   └── visualization.py  # Módulo con Plotting automático (Matplotlib y RDKit Draw).
│
├── train.py              # Script principal iterativo de ML: Configura el Optimizador y entrena epoch a epoch.
├── baseline.py           # ML Clásico con Scikit-learn (RandomForest y Morgan Fingerprint).
├── requirements.txt      # Requisitos de entorno (Torch, RdKit...).
└── README.md
```
