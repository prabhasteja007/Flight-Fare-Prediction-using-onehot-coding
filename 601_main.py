import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, confusion_matrix, classification_report
)

# -------------------------
# Stop mapping (consistent)
# -------------------------
stop_map = {'zero': 0, 'one': 1, 'two_or_more': 2}

# -----------------------------------
# Preprocessing Function
# -----------------------------------
def preprocess_data_improved(df):
    df = df.drop(['Unnamed: 0', 'flight'], axis=1)
    df['class'] = df['class'].apply(lambda x: 1 if x == 'Business' else 0)
    df['stops'] = df['stops'].map(stop_map)
    cat_cols = ['airline', 'source_city', 'destination_city', 'arrival_time', 'departure_time']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df['log_price'] = np.log1p(df['price'])
    return df

# -----------------------------------
# GUI Prediction Function
# -----------------------------------
def predict_price_gui():
    try:
        airline = airline_var.get()
        source = source_var.get()
        dest = dest_var.get()
        arrival = arrival_var.get()
        departure = departure_var.get()
        stops = stops_var.get()
        travel_class = 1 if class_var.get() == 'Business' else 0

        duration = float(duration_entry.get())
        days_left = int(days_left_entry.get())

        input_dict = {
            'duration': duration,
            'days_left': days_left,
            'class': travel_class,
            'stops': stop_map[stops]
        }

        for col in X.columns:
            if col.startswith('airline_'):
                input_dict[col] = 1 if col == f'airline_{airline}' else 0
            elif col.startswith('source_city_'):
                input_dict[col] = 1 if col == f'source_city_{source}' else 0
            elif col.startswith('destination_city_'):
                input_dict[col] = 1 if col == f'destination_city_{dest}' else 0
            elif col.startswith('arrival_time_'):
                input_dict[col] = 1 if col == f'arrival_time_{arrival}' else 0
            elif col.startswith('departure_time_'):
                input_dict[col] = 1 if col == f'departure_time_{departure}' else 0
            elif col not in input_dict:
                input_dict[col] = 0

        input_df = pd.DataFrame([input_dict])[X.columns]
        input_scaled = scaler.transform(input_df)
        input_poly = poly.transform(input_scaled)

        # Predict class
        class_pred = log_reg.predict(input_poly)[0]

        # Predict price based on class
        if class_pred == 0:
            log_price = lin_reg_cheap.predict(input_poly)[0]
        else:
            log_price = lin_reg_expensive.predict(input_poly)[0]

        final_price = np.expm1(log_price)

        result_label.config(text=f"Predicted Price: ₹{final_price:.2f}\n(MAE: ₹{mae_val:.2f}, RMSE: ₹{rmse_val:.2f})")

    except Exception as e:
        messagebox.showerror("Input Error", f"Error: {str(e)}")

# Function to create dropdown in GUI
def create_dropdown(label, var, options):
    frame = tk.Frame(root, bg="#f9f9f9")
    frame.pack(pady=5)
    tk.Label(frame, text=label, width=20, anchor='w', font=("Arial", 11), bg="#f9f9f9").pack(side='left')
    menu = ttk.Combobox(frame, textvariable=var, values=options, state="readonly", width=30)
    menu.current(0)
    menu.pack(side='right')

# Function to create entry field in GUI
def create_entry(label_text):
    frame = tk.Frame(root, bg="#f9f9f9")
    frame.pack(pady=5)
    tk.Label(frame, text=label_text, width=20, anchor='w', font=("Arial", 11), bg="#f9f9f9").pack(side='left')
    entry = tk.Entry(frame, width=32)
    entry.pack(side='right')
    return entry

if __name__ == "__main__":
    # -----------------------------------
    # Load and Prepare Data
    # -----------------------------------
    df_raw = pd.read_csv(r'C:\Users\tk896\OneDrive\Desktop\601\Project\Clean_Dataset.csv')

    # Dropdown values BEFORE preprocessing
    airlines = df_raw['airline'].unique().tolist()
    source_cities = df_raw['source_city'].unique().tolist()
    destination_cities = df_raw['destination_city'].unique().tolist()
    arrival_times = df_raw['arrival_time'].unique().tolist()
    departure_times = df_raw['departure_time'].unique().tolist()
    stops_options = list(stop_map.keys())
    classes = ['Economy', 'Business']

    # Preprocess
    df = preprocess_data_improved(df_raw.copy())

    # Features & Targets
    X = df.drop(columns=['price', 'log_price'])
    y_reg = df['log_price']
    y_class = (df['price'] > df['price'].median()).astype(int)

    # Scale and Polynomial Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    # Logistic Regression for classification
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_poly, y_class, test_size=0.2, random_state=42)
    log_reg = LogisticRegression(max_iter=2000)
    log_reg.fit(X_train_cls, y_train_cls)
    y_pred_cls = log_reg.predict(X_test_cls)
    y_prob_cls = log_reg.predict_proba(X_test_cls)[:, 1]

    print("\nLogistic Regression Metrics:")
    print("Accuracy:", accuracy_score(y_test_cls, y_pred_cls))
    print("Classification Report:\n", classification_report(y_test_cls, y_pred_cls, target_names=['<= Median', '> Median']))

    cm = confusion_matrix(y_test_cls, y_pred_cls)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['<= Median', '> Median'], yticklabels=['<= Median', '> Median'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Logistic Regression')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test_cls, y_prob_cls, alpha=0.5, color='green')
    plt.xlabel('Actual Class')
    plt.ylabel('Predicted Probability (> Median)')
    plt.title('Logistic Regression: Predicted Probability vs Actual Class')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Separate data for cheap and expensive flights
    cheap_mask = y_class == 0
    expensive_mask = y_class == 1

    X_cheap = X_poly[cheap_mask]
    y_cheap = y_reg[cheap_mask]

    X_expensive = X_poly[expensive_mask]
    y_expensive = y_reg[expensive_mask]

    # Train linear regressors
    lin_reg_cheap = LinearRegression()
    lin_reg_cheap.fit(X_cheap, y_cheap)
    y_cheap_pred = lin_reg_cheap.predict(X_cheap)
    
    # Calculate metrics for cheap flights model
    r2_cheap = r2_score(y_cheap, y_cheap_pred)
    mae_cheap = mean_absolute_error(y_cheap, y_cheap_pred)
    rmse_cheap = np.sqrt(mean_squared_error(y_cheap, y_cheap_pred))
    
    lin_reg_expensive = LinearRegression()
    lin_reg_expensive.fit(X_expensive, y_expensive)
    y_expensive_pred = lin_reg_expensive.predict(X_expensive)
    
    # Calculate metrics for expensive flights model
    r2_expensive = r2_score(y_expensive, y_expensive_pred)
    mae_expensive = mean_absolute_error(y_expensive, y_expensive_pred)
    rmse_expensive = np.sqrt(mean_squared_error(y_expensive, y_expensive_pred))
    
    # Print linear regression metrics
    print("\nLinear Regression Metrics for Cheap Flights (≤ Median):")
    print("R2 Score:", r2_cheap)
    print("MAE:", mae_cheap)
    print("RMSE:", rmse_cheap)
    
    print("\nLinear Regression Metrics for Expensive Flights (> Median):")
    print("R2 Score:", r2_expensive)
    print("MAE:", mae_expensive)
    print("RMSE:", rmse_expensive)

    # Evaluate on test set
    X_test_reg, y_test_reg = X_poly, df['price']

    # Predict class
    y_class_pred = log_reg.predict(X_test_reg)

    # Predict price based on class
    y_price_pred = []
    for i in range(len(X_test_reg)):
        if y_class_pred[i] == 0:
            log_price = lin_reg_cheap.predict([X_test_reg[i]])[0]
        else:
            log_price = lin_reg_expensive.predict([X_test_reg[i]])[0]
        price = np.expm1(log_price)
        y_price_pred.append(price)

    y_price_pred = np.array(y_price_pred)

    # Metrics for hybrid model
    mae_val = mean_absolute_error(y_test_reg, y_price_pred)
    rmse_val = np.sqrt(mean_squared_error(y_test_reg, y_price_pred))
    r2_val = r2_score(y_test_reg, y_price_pred)

    print("\nHybrid Model Metrics:")
    print("R2 Score:", r2_val)
    print("MAE:", mae_val)
    print("RMSE:", rmse_val)

    # Create a figure with subplots for visualizations
    plt.figure(figsize=(18, 12))
    
    # 1. Hybrid Model Predictions vs Actual
    plt.subplot(2, 2, 1)
    plt.scatter(y_test_reg, y_price_pred, alpha=0.4, color='orange')
    plt.plot([min(y_test_reg), max(y_test_reg)], [min(y_test_reg), max(y_test_reg)], 'r--')
    plt.xlabel('Actual Flight Price')
    plt.ylabel('Predicted Flight Price')
    plt.title("Hybrid Model: Prediction vs Actual Price")
    plt.grid(True)
    
    # 2. Residuals plot
    plt.subplot(2, 2, 2)
    residuals = y_test_reg - y_price_pred
    plt.scatter(y_price_pred, residuals, alpha=0.4, color='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    plt.grid(True)
    
    # 3. Distribution of predictions vs actuals
    plt.subplot(2, 2, 3)
    plt.hist(y_test_reg, bins=30, alpha=0.5, label='Actual Prices', color='blue')
    plt.hist(y_price_pred, bins=30, alpha=0.5, label='Predicted Prices', color='orange')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.title('Distribution of Actual vs Predicted Prices')
    plt.legend()
    plt.grid(True)
    
    # 4. Error distribution
    plt.subplot(2, 2, 4)
    percentage_error = (np.abs(y_test_reg - y_price_pred) / y_test_reg) * 100
    plt.hist(percentage_error, bins=20, color='purple', alpha=0.7)
    plt.xlabel('Percentage Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Percentage Error')
    plt.axvline(x=np.mean(percentage_error), color='r', linestyle='--', 
                label=f'Mean Error: {np.mean(percentage_error):.2f}%')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Feature importance visualization
    plt.figure(figsize=(12, 8))
    
    # Get feature importance from logistic regression
    feature_importance = np.abs(log_reg.coef_[0])
    # Get top 15 features
    top_indices = np.argsort(feature_importance)[-15:]
    
    # Create bar chart of feature importance
    plt.barh(range(len(top_indices)), feature_importance[top_indices], align='center')
    plt.yticks(range(len(top_indices)), [poly.get_feature_names_out()[i] for i in top_indices])
    plt.title('Top 15 Feature Importance in Logistic Regression Model')
    plt.xlabel('Coefficient Magnitude')
    plt.tight_layout()
    plt.show()
    
    # Visualize price by airline
    plt.figure(figsize=(14, 6))
    airline_prices = df_raw.groupby('airline')['price'].agg(['mean', 'std']).sort_values('mean')
    
    # Bar chart with error bars
    plt.bar(range(len(airline_prices)), airline_prices['mean'], yerr=airline_prices['std'], 
            capsize=5, alpha=0.7, color='skyblue')
    plt.xticks(range(len(airline_prices)), airline_prices.index, rotation=45, ha='right')
    plt.title('Average Flight Price by Airline with Standard Deviation')
    plt.xlabel('Airline')
    plt.ylabel('Average Price (₹)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # -----------------------------------
    # GUI Interface
    # -----------------------------------
    root = tk.Tk()
    root.title("Flight Price Predictor")
    root.geometry("600x750")
    root.configure(bg="#f9f9f9")

    # Header
    tk.Label(root, text="Flight Price Predictor", font=("Arial", 16, "bold"), bg="#f9f9f9").pack(pady=15)
    
    # Model metrics summary
    metrics_frame = tk.Frame(root, bg="#f0f0f0", padx=10, pady=10)
    metrics_frame.pack(fill='x', padx=20, pady=10)
    
    tk.Label(metrics_frame, text="Model Performance", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(anchor='w')
    tk.Label(metrics_frame, text=f"R² Score: {r2_val:.4f}", bg="#f0f0f0").pack(anchor='w')
    tk.Label(metrics_frame, text=f"MAE: ₹{mae_val:.2f}", bg="#f0f0f0").pack(anchor='w')
    tk.Label(metrics_frame, text=f"RMSE: ₹{rmse_val:.2f}", bg="#f0f0f0").pack(anchor='w')

    airline_var = tk.StringVar()
    source_var = tk.StringVar()
    dest_var = tk.StringVar()
    arrival_var = tk.StringVar()
    departure_var = tk.StringVar()
    stops_var = tk.StringVar()
    class_var = tk.StringVar()

    # Input section title
    tk.Label(root, text="Enter Flight Details", font=("Arial", 12, "bold"), bg="#f9f9f9").pack(pady=(15, 5))
    
    create_dropdown("Airline", airline_var, airlines)
    create_dropdown("Source City", source_var, source_cities)
    create_dropdown("Destination City", dest_var, destination_cities)
    create_dropdown("Arrival Time", arrival_var, arrival_times)
    create_dropdown("Departure Time", departure_var, departure_times)
    create_dropdown("Stops", stops_var, stops_options)
    create_dropdown("Class", class_var, classes)

    duration_entry = create_entry("Duration (hours)")
    days_left_entry = create_entry("Days Left")

    # Stylized prediction button
    predict_button = tk.Button(
        root, 
        text="Predict Price", 
        command=predict_price_gui,
        bg="#4CAF50", 
        fg="white", 
        font=("Arial", 12, "bold"),
        padx=20,
        pady=10,
        relief=tk.RAISED,
        borderwidth=2
    )
    predict_button.pack(pady=20)

    # Results section with border and styling
    result_frame = tk.Frame(root, bg="#e8f4ea", padx=15, pady=15, relief=tk.GROOVE, borderwidth=2)
    result_frame.pack(padx=20, pady=10, fill='x')
    
    tk.Label(result_frame, text="Prediction Result", font=("Arial", 12, "bold"), bg="#e8f4ea").pack()
    
    result_label = tk.Label(result_frame, text="", font=("Arial", 14), fg="blue", bg="#e8f4ea")
    result_label.pack(pady=10)

    root.mainloop()
