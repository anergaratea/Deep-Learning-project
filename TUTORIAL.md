# Tutorial didáctico: Del SMILES al vector de propiedades moleculares con GNN

Este tutorial explica, paso a paso y desde una perspectiva didáctica (asumiendo nociones básicas de Deep Learning), cómo este repo toma moléculas en formato SMILES, las convierte en grafos, y entrena una Red Neuronal de Grafos (GNN) moderna para predecir propiedades moleculares.

---

## 1. Preprocesamiento y Conversión de datos

**¿Por qué?**
Las moléculas pueden representarse como cadenas SMILES, pero para que una GNN opere, necesitamos trabajar con grafos: átomos como nodos y enlaces como aristas. El preprocesamiento convierte SMILES en una estructura de grafo con características químico-estructurales.

**¿Cómo?**
- El script `data/preprocessing.py` toma SMILES y genera, para cada molécula,:
  - Matriz de características de átomos (`atom_features`)
  - Matriz de aristas (`edges` y `edge_features`)
  - Etiqueta numérica (propiedad a predecir)

Esto permite alimentar eficientemente la GNN. 

---

## 2. Construcción y entrenamiento del modelo GNN

### ¿Qué es una GNN y por qué se usa aquí?
Una Graph Neural Network permite procesar información estructurada y relacional (por ejemplo, la topología molecular). Cada nodo aprende no solo de sus propios atributos, sino también de los de los nodos vecinos (propagación de mensajes). Esto ayuda a extraer representaciones más ricas para tareas de predicción.

### Pipeline básico de entrenamiento (`train.py`)

1. **Carga del dataset:**
   - Llama a `load_real_dataset()` para obtener SMILES y valores reales.
   - Convierte los datos en grafos con `create_dataset()`.

2. **Split y generación de lotes:**
   - Se reparten los datos en entrenamiento y validación.
   - Se utiliza un `DataLoader` customizado (`collate_fn`) para procesar eficientemente lotes con diferentes tamaños de grafos.

3. **Selección del modelo:**
   - Puedes elegir entre `MolecularGNN` (convencional de message-passing) o `MolecularGAT` (atención sobre nodos vecinos importantes). Cambia la variable `use_gat` en `train.py`.

4. **Entrenamiento:**
   - En cada epoch, la red procesa lotes de moléculas, propaga la información estructural y ajusta los pesos para minimizar el error de predicción. Se imprime la pérdida (Loss) cada 10 epochs.

5. **Evaluación:**
   - Se calcula el error en el set de validación, permitiendo supervisar overfitting y generalización.

---

## 3. Interpretación de métricas y visualizaciones

- **`training_metrics.png`**: muestra la evolución de la loss de entrenamiento y validación.
    - Si ambas disminuyen y la de validación no sube abruptamente, el modelo está generalizando bien.
    - Si la de entrenamiento baja pero la de validación sube, puede haber overfitting.
- **`true_vs_predicted.png`**: scatter plot del valor real frente al predicho. Una nube cercana a la diagonal representa buen ajuste.
- **`molecule.png`**: ejemplo visual de una molécula procesada.

---

## 4. Experimenta tú mismo

```bash
# Prepara entorno y dependencias
pip install -r requirements.txt

# Entrena el modelo GNN (o GAT)
python train.py
```

Modifica parámetros relevantes en `train.py` para experimentar: tamaño de capas, número de épocas, activar GAT, etc. Puedes observar cómo cambian las métricas y los gráficos.

---

## 5. ¿Dónde aprender más?
- [Introducción a GNNs (Distill.pub)](https://distill.pub/2021/gnn-intro/)
- [El paper base en NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/8b7d1512c1235d0c4dd3ce7d7d1be29f-Abstract.html)
- [RDKit para representación molecular](https://www.rdkit.org/)

---

¿Dudas? Lee los comentarios en los scripts (especialmente `train.py`), busca términos en este tutorial, o revisa el README para ejemplos ejecutables. ¡Explora, modifica y aprende practicando!
