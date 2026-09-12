# master-thesis

## `generate_random_matrix` — bug de conectividad

`generate_random_matrix(n_concepts, density)` genera la matriz de influencia `W` de un FCM
(diagonal cero, valores no nulos en `[-1, 1]`, ~`density * n²` elementos no nulos) y pretende
garantizar que ningún nodo quede completamente desconectado (al menos una conexión entrante y
una saliente por nodo).

**Bug conocido:** la garantía de conectividad **no se cumple en densidades bajas**. La
reparación solo se ejecuta cuando la fila **y** la columna de un nodo están ambas completamente
a cero (`if fila == 0 and columna == 0`); un nodo con solo una de las dos vacía no se arregla,
y en densidades bajas es frecuente terminar con filas o columnas vacías.

Afecta a ambas implementaciones:

- `leonardo/random_matrix.py` (numpy, original)
- `core/utils.generate_random_matrix` (port PyTorch, mismo comportamiento)

No se ha corregido a propósito: el port reproduce fielmente el comportamiento original y una
corrección cambiaría los resultados respecto a la referencia.
