import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import io
from sklearn.metrics import pairwise_distances_argmin_min


# ------------------ Cargar los modelos y el train set ------------------
# Asegurarse de que la sesion empieza en estado Default
if "predicted" not in st.session_state:
    st.session_state.predicted = False
    st.session_state.input_values = None
    st.session_state.pred_encoded = None
    st.session_state.proba_encoded = None

BRAND_LIST = [
    "maruti",
    "hyundai",
    "mahindra",
    "tata",
    "honda",
    "ford",
    "toyota",
    "chevrolet",
    "volkswagen",
    "renault",
    "skoda",
    "audi",
    "nissan",
    "fiat",
    "mercedes-benz",
    "bmw",
    "datsun",
    "land",
    "jaguar",
    "mitsubishi",
    "volvo",
    "ambassador",
    "jeep",
    "opelcorsa",
    "kia",
    "mg",
    "daewoo",
    "force",
    "isuzu"
]

# Cortar la marca del nombre
def extract_brand_from_name(name: str) -> str:
    name_lower = name.strip().lower()
    for b in BRAND_LIST:
        if name_lower.startswith(b):
            return b
    return "other"

# Cargar el training data para visualizacion
@st.cache_data(show_spinner=False)
def load_training_data():
    df = pd.read_csv("data/cars_data.csv")
    
    # Extraer la columna de "brand" 
    df["brand"] = df["name"].apply(extract_brand_from_name)

    # Convertir la columna a la categoria
    df["brand"] = df["brand"].astype("category")

    return df

df_train = load_training_data()

# Definir las features exactas que el modelo usa
raw_features = [
    "year",
    "selling_price",
    "transmission",
    "tipo_carroceria",
    "potencia_motor_hp",
    "nivel_seguridad",
    "eficiencia_km_l",
    "brand"
]

# Cargar el pipeline entrenado
@st.cache_data(show_spinner=False)
def load_models():
    pipeline = joblib.load("models/xgb_final_pipeline.pkl")
    le = joblib.load("models/label_encoder.pkl")
    return pipeline, le

model, le = load_models()

# Titulo y descripcion
st.set_page_config(page_title="Car Quality Predictor", layout="centered")
st.title("Car Quality Predictor by Lakitus")
st.markdown(
    """
    Fill in the sidebar with the car's specifications and click **Predict**
    """
)


# ------------------ Configuracion de las features en el sidebar ------------------
st.sidebar.header("Base Car Specifications")

year = st.sidebar.number_input("Year", min_value=1980, max_value=2025, value=2015, step=1)
selling_price = st.sidebar.number_input("Selling Price (MXN)", min_value=0, max_value=10_000_000, value=200000, step=1000)
# km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=500_000, value=50_000, step=1000)

# Codigos categoricos
# fuel = st.sidebar.selectbox("Fuel Type (code)", options={
#     "Petrol (0)":0, "Diesel (1)":1, "CNG (2)":2, "LPG (3)":3, "Electric (4)":4
# })
# comb_est = st.sidebar.number_input("Combustible Estimado (L)", min_value=0.0, max_value=10000.0, value=2000.0, step=100.0)
# seller_type = st.sidebar.selectbox("Seller Type (code)", options={
#     "Dealer (0)":0, "Individual (1)":1, "Trustmark(2)":2
# })
transmission = st.sidebar.selectbox("Transmission (code)", options={"Manual (0)":0, "Automatic (1)":1})
# owner = st.sidebar.selectbox("Owner Type (code)", options={
#     "First Owner (0)":0, "Second Owner (1)":1, "Third Owner (2)":2, "Fourth & Above (3)":3, "Test Drive (4)":4
# })
tipo_carroceria = st.sidebar.selectbox("Body Type (code)", options={
    "Sedan (1)":1, "Hatchback (2)":2, "SUV (3)":3, "Pickup (4)":4, "Sport (5)":5
})
potencia_motor_hp = st.sidebar.number_input("Engine Power (hp)", min_value=10, max_value=500, value=100, step=10)
nivel_seguridad = st.sidebar.number_input("Safety Rating", min_value=0.0, max_value=5.0, value=3.0, step=0.1)
eficiencia_km_l = st.sidebar.number_input("Efficiency (km/l)", min_value=1.0, max_value=30.0, value=18.0, step=0.5)

brand = st.sidebar.selectbox(
    "Brand", BRAND_LIST
)


# ------------------ Construccion de dataframes y diccionarios ------------------
# Construir un dataframe de el input generado por el usuario
input_dict = {
    "year": [year],
    "selling_price": [selling_price],
    # "km_driven": [km_driven],
    # "fuel": [fuel],
    # "combustible_estimado_l": [comb_est],
    # "seller_type": [seller_type],
    "transmission": [transmission],
    # "owner": [owner],
    "tipo_carroceria": [tipo_carroceria],
    "potencia_motor_hp": [potencia_motor_hp],
    "nivel_seguridad": [nivel_seguridad],
    "eficiencia_km_l": [eficiencia_km_l],
    "brand": [brand]
}

# Inputs cortados para predicciones cercanas
df_current = pd.DataFrame({
    "year":               [year],
    "selling_price":      [selling_price],
    "transmission":       [transmission],
    "tipo_carroceria":    [tipo_carroceria],
    "potencia_motor_hp":  [potencia_motor_hp],
    "nivel_seguridad":    [nivel_seguridad],
    "eficiencia_km_l":    [eficiencia_km_l],
    "brand":              [brand]    
})

# Convertir features categoricas
for col in ["transmission", "tipo_carroceria"]:
    df_current[col] = df_current[col].astype("category")

# Leer los inputs del side bar para guardarlos cuando se use "Predict"
current_inputs = {
    "year": year,
    "selling_price": selling_price,
    # "km_driven": km_driven,
    # "fuel": fuel,
    # "combustible_estimado_l": comb_est,
    # "seller_type": seller_type,
    "transmission": transmission,
    # "owner": owner,
    "tipo_carroceria": tipo_carroceria,
    "potencia_motor_hp": potencia_motor_hp,
    "nivel_seguridad": nivel_seguridad,
    "eficiencia_km_l": eficiencia_km_l,
    "brand": brand
}
input_df = pd.DataFrame(input_dict)


# ------------------ Implementacion del boton de prediccion ------------------
if st.sidebar.button("🔍 Predict"):
    # Guardar los inputs en session state
    st.session_state.input_values = current_inputs.copy()
    st.session_state.predicted = True

    # Construir un data frame de una sola fila de los inputs guardados
    df = pd.DataFrame({k: [v] for k, v in st.session_state.input_values.items()})
    for col in ["transmission", "tipo_carroceria", "brand"]: # add "brand"
        df[col] = df[col].astype("category")

    # Crear la prediccion inicial codificada
    st.session_state.pred_encoded = model.predict(df)
    st.session_state.proba_encoded = model.predict_proba(df)[0]


# ------------------ Mostrtar predicciones + SHAP + sliders ------------------
if st.session_state.predicted:
    # Recoinstruir input del session state
    base_df = pd.DataFrame({k: [v] for k, v in st.session_state.input_values.items()})
    for col in ["transmission", "tipo_carroceria", "brand"]: # add "brand"
        base_df[col] = base_df[col].astype("category")

    # Decodificar las labels y construir el diccionario de probabilidades
    pred_label = le.inverse_transform(st.session_state.pred_encoded)[0]
    classes = le.classes_.tolist()
    proba_dict = {classes[i]: round(float(st.session_state.proba_encoded[i]), 3)
                  for i in range(len(classes))}


    # ------------------ Visualizacion de la prediccion del sidebar ------------------
    # --- prettier prediction label ---
    color_map = {
    "Baja": "#E63946",   # red
    "Media": "#F4A261",  # orange
    "Alta": "#0ED450"    # green
    }
    label_color = color_map.get(pred_label, "#FFFFFF")

    st.markdown(
        f"<h3 style='text-align:center; color:white;'>{'Predicted Quality'}</h3>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<h1 style='text-align:center; color:{label_color};'>{pred_label}</h1>",
        unsafe_allow_html=True
    )
    # st.write("**Class Probabilities:**")
    # st.json(proba_dict)

    # Mostrar la probabilidad de cada clase de esa prediccion
    # st.subheader("Class Probabilities")
    pred_confidence = proba_dict[pred_label]  
    st.write("**Prediction Confidence**")
    # st.progress expects a value between 0.0 and 1.0
    st.progress(pred_confidence)

    # 8. Below it, show the class probabilities in metric boxes
    st.write("**Class Probabilities**")
    col1, col2, col3 = st.columns(3)
    col1.metric("P(Alta)", f"{proba_dict['Alta']:.2f}")
    col2.metric("P(Media)", f"{proba_dict['Media']:.2f}")
    col3.metric("P(Baja)", f"{proba_dict['Baja']:.2f}")


    # ------------------ Visualizacion de la grafica SHAP ------------------
    st.write("")  # para el spacing
    st.subheader("SHAP Feature Contributions")

    # Transformar el input usando el preprocessador del pipeline
    preprocessor = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]
    X_transformed = preprocessor.transform(base_df)  # shape: (1, n_features)

    # Inicializar TreeExplainer en el XGBClassifier entrenado
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed)

    # For multiclass, shap_values is a list of arrays (one per class).
    # We'll show the SHAP values for the predicted class:
    # pred_class_index = list(clf.classes_).index(pred_encoded[0])

    # Fix
    # shap_vals_for_pred = shap_values[pred_class_index][0]  # shape: (n_features,)
    # Now adjust based on the shape/type of shap_values:
    if isinstance(shap_values, list):
        # The old style: shap_values is a Python list of length n_classes,
        # each entry is an array of shape (n_samples, n_features).
        # For our single sample, use shap_values[pred_class_index][0].
        shap_vals_for_pred = shap_values[st.session_state.pred_encoded[0]][0]
    else:
        # The new style: shap_values is a single NumPy array of shape (n_samples, n_features, n_classes).
        # For sample 0, pick the column for pred_class_index:
        pred_class_index = list(clf.classes_).index(st.session_state.pred_encoded[0])
        shap_vals_for_pred = shap_values[0, :, pred_class_index]

    # Obtener los nombres de las features del preprocessor
    feature_names = preprocessor.get_feature_names_out()

    # Construir un DataFrame de features vs. SHAP values
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_vals_for_pred
    })
    shap_df["abs_shap"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values(by="abs_shap", ascending=False)

    # Mostrar los top-10 SHAP features
    top_n = 10
    top_shap = shap_df.head(top_n).sort_values(by="shap_value")
    fig, ax = plt.subplots(figsize=(6, 4))

    # ---> DARK-MODE plotting starts here:
    fig.patch.set_facecolor("#0f1116")
    ax.set_facecolor("#0f1116")
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.set_xlabel("SHAP Value", color="white")
    ax.set_ylabel("Feature", color="white")
    ax.set_title(f"Top {top_n} SHAP Features for '{pred_label}'", color="white")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    # ---> DARK-MODE ends here

    ax.barh(
        top_shap["feature"],
        top_shap["shap_value"],
        color=["#1f77b4" if v > 0 else "#ff7f0e" for v in top_shap["shap_value"]]
    )
    st.pyplot(fig)


    # ------------------ Boton de descarga del SHAP plot ------------------
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    st.download_button(
        label="Download SHAP Plot as PNG",
        data=buf,
        file_name="shap_plot.png",
        mime="image/png"
    )
    
    # ------------------ Mostrar "Show Nearby Training Examples" ------------------
    st.write("")  # spacing

    # Cortar el dataframe a ser entrenado
    df_train_trimmed = df_train[ raw_features + ["calidad_auto"] ].copy()
    X_train_feats = df_train_trimmed[raw_features].copy()
    for col in ["transmission", "tipo_carroceria", "brand"]:
        X_train_feats[col] = X_train_feats[col].astype("category")

    @st.cache_data
    def get_transformed_training(feats_df):
        return preprocessor.transform(feats_df)

    X_train_trans = get_transformed_training(X_train_feats)

    X_curr_feats = df_current[raw_features].copy()
    for col in ["transmission", "tipo_carroceria", "brand"]:
        X_curr_feats[col] = X_curr_feats[col].astype("category")

    X_curr_trans = preprocessor.transform(X_curr_feats)
    idx, _ = pairwise_distances_argmin_min(X_curr_trans, X_train_trans)
    nearby_row = df_train_trimmed.iloc[idx[0]]
    st.subheader("Nearest Training-Set Example")
    st.write(nearby_row)
    

    # ------------------ Mostrar los What-If sliders ------------------
    st.write("")  # spacing
    st.write("---")
    st.subheader("🛠 What-If Sliders")

    # Leer valores originales del session_state
    orig = st.session_state.input_values

    # 1) Year slider
    new_year = st.slider(
        "Year",
        min_value=1980, max_value=2025, value=orig["year"], step=1
    )

    # 2) Engine Power (hp) slider
    new_pot = st.slider(
        "Engine Power (hp)",
        min_value=10, max_value=500, value=orig["potencia_motor_hp"], step=10
    )

    # 3) Safety Rating slider
    new_nivel = st.slider(
        "Safety Rating (nivel_seguridad)",
        min_value=0.0, max_value=5.0, value=orig["nivel_seguridad"], step=0.1
    )

    # 4) Efficiency slider
    new_efic = st.slider(
        "Efficiency (km/l)",
        min_value=0.0, max_value=30.0, value=orig["eficiencia_km_l"], step=0.5
    )

    # Reconstruir un nuevo data frame que sobreescriba solo estos valores
    df2 = base_df.copy()
    df2["year"] = new_year
    df2["potencia_motor_hp"] = new_pot
    df2["nivel_seguridad"] = new_nivel
    df2["eficiencia_km_l"] = new_efic

    # Re castear las categorias
    for col in ["transmission", "tipo_carroceria", "brand"]: # add "brand"
        df2[col] = df2[col].astype("category")

    # Hacer una nueva prediccion en df2
    pred2_enc = model.predict(df2)
    pred2_label = le.inverse_transform(pred2_enc)[0]
    proba2_enc = model.predict_proba(df2)[0]
    proba2_dict = {classes[i]: round(float(proba2_enc[i]), 3) for i in range(len(classes))}


    # ------------------ Visualizacion de las What-If predictions ------------------
    label2_color = color_map.get(pred2_label, "#FFFFFF")
    st.markdown(
        f"<h3 style='text-align:center; color:white;'>{'What-If Quality'}</h3>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<h1 style='text-align:center; color:{label2_color};'>{pred2_label}</h1>",
        unsafe_allow_html=True
    )

    pred2_confidence = proba2_dict[pred2_label]  
    st.write("**Prediction Confidence**")
    # st.progress expects a value between 0.0 and 1.0
    st.progress(pred2_confidence)

    # 8. Below it, show the class probabilities in metric boxes
    st.write("**Class Probabilities**")
    col1, col2, col3 = st.columns(3)
    col1.metric("P(Alta)", f"{proba2_dict['Alta']:.2f}")
    col2.metric("P(Media)", f"{proba2_dict['Media']:.2f}")
    col3.metric("P(Baja)", f"{proba2_dict['Baja']:.2f}")


    # ------------------ Boton de “Download Prediction” ------------------
    result_df = input_df.copy()
    result_df["predicted_quality"] = pred_label
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Prediction as CSV",
        data=csv,
        file_name="car_quality_prediction.csv",
        mime="text/csv"
    )


else:
    st.write("Adjust the inputs in the sidebar, then click **Predict**.")
