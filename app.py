import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Asegurarse de que la sesion empieza en estado Default
if "predicted" not in st.session_state:
    st.session_state.predicted = False
    st.session_state.input_values = None
    st.session_state.pred_encoded = None
    st.session_state.proba_encoded = None

# Cargar el pipeline entrenado
@st.cache_data(show_spinner=False)
def load_models():
    pipeline = joblib.load("models/xgb_final_pipeline.pkl")
    le = joblib.load("models/label_encoder.pkl")
    return pipeline, le

model, le = load_models()

# Titulo y descripcion
st.set_page_config(page_title="Car Quality Predictor", layout="centered")
st.title("🚗 Car Quality Predictor")
st.markdown(
    """
    Enter the car’s specifications below and click **Predict** to see if the car quality is **Alta**, **Media**, or **Baja**.
    """
)

# Main form para inputs de usuario
st.sidebar.header("Car Specifications")

year = st.sidebar.number_input("Year", min_value=1980, max_value=2025, value=2015, step=1)
selling_price = st.sidebar.number_input("Selling Price", min_value=0, max_value=1_000_000, value=200000, step=1000)
km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=500_000, value=50_000, step=1000)

# Codigos categoricos
fuel = st.sidebar.selectbox("Fuel Type (code)", options={
    "Petrol (0)":0, "Diesel (1)":1, "CNG (2)":2, "LPG (3)":3, "Electric (4)":4
})
comb_est = st.sidebar.number_input("Combustible Estimado (L)", min_value=0.0, max_value=10000.0, value=2000.0, step=100.0)
seller_type = st.sidebar.selectbox("Seller Type (code)", options={
    "Dealer (0)":0, "Individual (1)":1, "Trustmark(2)":2
})
transmission = st.sidebar.selectbox("Transmission (code)", options={"Manual (0)":0, "Automatic (1)":1})
owner = st.sidebar.selectbox("Owner Type (code)", options={
    "First Owner (0)":0, "Second Owner (1)":1, "Third Owner (2)":2, "Fourth & Above (3)":3, "Test Drive (4)":4
})
tipo_carroceria = st.sidebar.selectbox("Body Type (code)", options={
    "Hatchback (1)":1, "Sedan (2)":2, "MUV/SUV (3)":3, "Pickup (4)":4, "Van (5)":5
})
potencia_motor_hp = st.sidebar.number_input("Engine Power (hp)", min_value=10, max_value=1000, value=100, step=10)
nivel_seguridad = st.sidebar.number_input("Safety Rating", min_value=0.0, max_value=5.0, value=3.0, step=0.1)
eficiencia_km_l = st.sidebar.number_input("Efficiency (km/l)", min_value=0.0, max_value=50.0, value=18.0, step=0.5)

# Usar en caso de que brand este disponible
# brand = st.sidebar.selectbox("Brand", options=[
#     "Maruti", "Hyundai", "Toyota", "Honda", "Ford", "Other"
# ])

# Construir un dataframe de el input generado por el usuario
input_dict = {
    "year": [year],
    "selling_price": [selling_price],
    "km_driven": [km_driven],
    "fuel": [fuel],
    "combustible_estimado_l": [comb_est],
    "seller_type": [seller_type],
    "transmission": [transmission],
    "owner": [owner],
    "tipo_carroceria": [tipo_carroceria],
    "potencia_motor_hp": [potencia_motor_hp],
    "nivel_seguridad": [nivel_seguridad],
    "eficiencia_km_l": [eficiencia_km_l]
    # "brand": [brand]
}
# input_df = pd.DataFrame(input_dict)

# # Asegurarse de que los dtypes categoricos igualen a los del training
# for col in ["fuel", "seller_type", "transmission", "owner", "tipo_carroceria"]: # Add "brand"
#     input_df[col] = input_df[col].astype("category")

# Leer los inputs del side bar para guardarlos cuando se use "Predict"
current_inputs = {
    "year": year,
    "selling_price": selling_price,
    "km_driven": km_driven,
    "fuel": fuel,
    "combustible_estimado_l": comb_est,
    "seller_type": seller_type,
    "transmission": transmission,
    "owner": owner,
    "tipo_carroceria": tipo_carroceria,
    "potencia_motor_hp": potencia_motor_hp,
    "nivel_seguridad": nivel_seguridad,
    "eficiencia_km_l": eficiencia_km_l
    # "brand": brand
}

# Implementacion del boton de prediccion
if st.sidebar.button("🔍 Predict"):
    # Guardar los inputs en session state
    st.session_state.input_values = current_inputs.copy()
    st.session_state.predicted = True

    # Construir un data frame de una sola fila de los inputs guardados
    df = pd.DataFrame({k: [v] for k, v in st.session_state.input_values.items()})
    for col in ["fuel", "seller_type", "transmission", "owner", "tipo_carroceria"]: # add "brand"
        df[col] = df[col].astype("category")

    # Crear la prediccion inicial codificada
    st.session_state.pred_encoded = model.predict(df)
    st.session_state.proba_encoded = model.predict_proba(df)[0]

# Si ya habiamos presionado "Predict" antes mostrtar los resultados + SHAP + sliders
if st.session_state.predicted:
    # Recoinstruir input del session state
    base_df = pd.DataFrame({k: [v] for k, v in st.session_state.input_values.items()})
    for col in ["fuel", "seller_type", "transmission", "owner", "tipo_carroceria"]: # add "brand"
        base_df[col] = base_df[col].astype("category")

    # Decodificar las labels y construir el diccionario de probabilidades
    pred_label = le.inverse_transform(st.session_state.pred_encoded)[0]
    classes = le.classes_.tolist()
    proba_dict = {classes[i]: round(float(st.session_state.proba_encoded[i]), 3)
                  for i in range(len(classes))}

    st.write("---")
    st.subheader("Prediction Result")
    st.write(f"**Predicted Quality:** ➜ **{pred_label}**")
    st.write("**Class Probabilities:**")
    st.json(proba_dict)

        # --- NEW: SHAP EXPLANATION ---
    st.write("")  # para el spacing
    st.subheader("🔎 SHAP Feature Contributions")

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

    # 5. Construir un DataFrame de features vs. SHAP values
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
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
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

    # --- WHAT-IF: MULTIPLE SLIDERS ---
    st.write("")  # spacing
    st.subheader("🛠 What-If: Vary Multiple Inputs")

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
        min_value=10, max_value=1000, value=orig["potencia_motor_hp"], step=10
    )

    # 3) Safety Rating slider
    new_nivel = st.slider(
        "Safety Rating (nivel_seguridad)",
        min_value=0.0, max_value=5.0, value=orig["nivel_seguridad"], step=0.1
    )

    # 4) Efficiency slider
    new_efic = st.slider(
        "Efficiency (km/l)",
        min_value=0.0, max_value=50.0, value=orig["eficiencia_km_l"], step=0.5
    )

    # Reconstruir un nuevo data frame que sobreescriba solo estos valores
    df2 = base_df.copy()
    df2["year"] = new_year
    df2["potencia_motor_hp"] = new_pot
    df2["nivel_seguridad"] = new_nivel
    df2["eficiencia_km_l"] = new_efic

    # Re‐cast categories (type unchanged for categorical columns)
    for col in ["fuel", "seller_type", "transmission", "owner", "tipo_carroceria"]: # add "brand"
        df2[col] = df2[col].astype("category")

    # Make a new prediction on df2
    pred2_enc = model.predict(df2)
    pred2_label = le.inverse_transform(pred2_enc)[0]
    proba2_enc = model.predict_proba(df2)[0]
    proba2_dict = {classes[i]: round(float(proba2_enc[i]), 3) for i in range(len(classes))}

    st.write(f"• With modified inputs, predicted quality ➜ **{pred2_label}**")
    st.write("• Probabilities:")
    st.json(proba2_dict)

else:
    st.write("Adjust the inputs in the sidebar, then click **Predict**.")
