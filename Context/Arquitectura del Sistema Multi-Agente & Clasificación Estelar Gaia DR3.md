**Proyecto:** Estancia IA-UNAM | MCE CIMAT Monterrey

**Objetivo:** Clasificación espectral automatizada mediante la integración de parámetros físicos, cinemática y aprendizaje profundo sobre espectros XP.

---

## 1. Capa de Datos (Data Sources)

Para un modelo de vanguardia, el sistema debe consumir tres tipos de fuentes:

- **Catálogo Maestro (Actual):** Parámetros estelares derivados (`teff`, `logg`, `mh`, `alphafe`, `ew_halpha`).
    
- **Astrometría y Cinemática:** Paralaje y movimientos propios (`parallax`, `pmra`, `pmdec`, `radial_velocity`).
    
- **Espectroscopía de Baja Resolución:** Coeficientes de los espectros BP/RP (XP Coefficients).
    

---

## 2. Estructura de Agentes Especialistas

Cada agente procesa una dimensión del dato y emite un veredicto en formato JSON.

### A. Agentes de Reglas Físicas (Basados en Conocimiento)

1. **Agente Térmico ($T_{eff}$):** Define el tipo espectral (O, B, A, F, G, K, M) mediante umbrales de temperatura.
    
2. **Agente de Gravedad ($log \ g$):** Determina la clase de luminosidad (Enana, Gigante, Supergigante).
    
3. **Agente Químico ($[M/H]$ y $[\alpha/Fe]$):** Identifica la población galáctica (Disco delgado, Disco grueso, Halo).
    
4. **Agente de Actividad ($H\alpha$):** Detecta líneas de emisión que sugieren juventud estelar o fenómenos exóticos.
    

### B. Agentes de Aprendizaje Estadístico (Ambiociosos)

5. **Agente Espectral (Deep Learning):** Una **Red Neuronal (CNN/Transformer)** que analiza los coeficientes XP para capturar rasgos sutiles no tabulados.
    
6. **Agente Cinemático (Dinámica):** Calcula velocidades espaciales $(U, V, W)$ para validar la coherencia entre la edad física de la estrella y su órbita galáctica.
    

---

## 3. Formato de Intercambio (JSON Consolidado)

El sistema integra las salidas en un objeto único que sirve de contexto para el LLM:

JSON

```
{
  "metadata": { "source_id": "int64", "coordenadas": [ra, dec] },
  "evidencias": {
    "fisica_estelar": {
      "tipo_sugerido": "G",
      "luminosidad": "V",
      "confianza": 0.95
    },
    "quimica": {
      "metalicidad": "Solar",
      "poblacion": "Disco Delgado"
    },
    "espectro_xp": {
      "prediccion_cnn": "G2V",
      "anomalias_detectadas": false
    },
    "cinematica": {
      "v_dispersion": "baja",
      "coherente_con_edad": true
    }
  }
}
```

---

## 4. Capa de Inteligencia y Auditoría (LLM + RAG)

Esta es la fase final donde el modelo de lenguaje actúa como un "Investigador Principal":

1. **Auditoría:** El LLM recibe el JSON y busca discrepancias (ej. una estrella que parece vieja químicamente pero joven cinemáticamente).
    
2. **RAG (Retrieval Augmented Generation):** El modelo consulta una base de conocimientos técnica (papers de Gaia, manuales de clasificación espectral) para fundamentar su diagnóstico.
    
3. **Output:** Genera una ficha técnica explicativa en lenguaje natural para el astrónomo.
    

---

## 5. Próximos Pasos Técnicos

- [ ] Implementar el pipeline de limpieza y normalización para los parámetros de GSP-Phot.
    
- [ ] Diseñar el prompt "System" para el LLM que definirá las reglas de auditoría.
    
- [ ] **Hito Ambicioso:** Descargar una muestra de `xp_continuous_mean_spectrum` para entrenar el Agente Espectral.
    

---

**¿Te gustaría que empezáramos a prototipar el cód**