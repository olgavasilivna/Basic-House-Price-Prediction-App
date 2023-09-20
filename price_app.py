import tkinter as tk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

class HousePricePredictionApp:
    def __init__(self, root):
        # Application window
        self.root = root
        self.root.title("Price App")
        
        #Button to load a CSV dataset
        self.load_data_button = tk.Button(root, text="Load CSV Dataset", command=self.load_csv)
        self.load_data_button.pack()
        
        # Label to select fields for training
        self.fields_label = tk.Label(root, text="Select Fields for Training:")
        self.fields_label.pack()
        
        # List with dataset columns
        self.fields_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE)
        self.fields_listbox.pack()
        
        # Button for model training
        self.train_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.train_button.pack()
        
    # Fields for user inputs:
        self.zipcode_label = tk.Label(root, text="Zip Code:")
        self.zipcode_label.pack()
        
        self.zipcode_entry = tk.Entry(root)
        self.zipcode_entry.pack()
        
        self.square_meters_label = tk.Label(root, text="House Size:")
        self.square_meters_label.pack()
        
        self.square_meters_entry = tk.Entry(root)
        self.square_meters_entry.pack()
        
        self.acre_size_label = tk.Label(root, text="Acre Lot:")
        self.acre_size_label.pack()
        
        self.acre_size_entry = tk.Entry(root)
        self.acre_size_entry.pack()
        
        self.bedrooms_label = tk.Label(root, text="Bedrooms:")
        self.bedrooms_label.pack()
        
        self.bedrooms_entry = tk.Entry(root)
        self.bedrooms_entry.pack()
        
        self.bathrooms_label = tk.Label(root, text="Bathrooms:")
        self.bathrooms_label.pack()
        
        self.bathrooms_entry = tk.Entry(root)
        self.bathrooms_entry.pack()
        
        # Button to predict the price
        self.predict_button = tk.Button(root, text="Predict Price", command=self.predict_price)
        self.predict_button.pack()
        
        # Initializing model and data variable
        self.model = None
        self.data = None
        self.X_test = None  
        self.y_test = None 

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                # Populate the list with column names from the dataset
                self.fields_listbox.delete(0, tk.END)
                for column in self.data.columns:
                    self.fields_listbox.insert(tk.END, column)
            except Exception as e:
                tk.messagebox.showerror("Error", f"Error loading data: {str(e)}")

    def train_model(self):
        selected_fields = self.fields_listbox.curselection()
        if not selected_fields:
            tk.messagebox.showwarning("Warning", "Select at least one field for training.")
            return

        selected_columns = [self.data.columns[i] for i in selected_fields]
        X = self.data[selected_columns]

        try:
            y = self.data['price']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Training using Random Forest Regressor
            self.model = RandomForestRegressor(random_state=42)  # You can adjust hyperparameters here
            self.model.fit(self.X_train, self.y_train)
            joblib.dump(self.model, 'house_price_model.pkl')

            # R2 score
            r2 = r2_score(self.y_test, self.model.predict(self.X_test))
            tk.messagebox.showinfo("Info", f"Model trained successfully.\nR2 Score: {r2:.2f}")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Error training model: {str(e)}")


    def predict_price(self):
        if not self.model:
            tk.messagebox.showwarning("Warning", "Train a model before making predictions.")
            return

        try:
            # Getting the features used for training the model
            selected_fields = self.fields_listbox.curselection()
            if not selected_fields:
                tk.messagebox.showwarning("Warning", "Select at least one field for training.")
                return

            selected_columns = [self.data.columns[i] for i in selected_fields]

            # Collecting input data 
            input_data = {column: [float(self.data[column].iloc[0])] for column in selected_columns}

            # Creating a df with feature names
            input_df = pd.DataFrame(input_data)

            # Predicting the price using the model
            predicted_price = self.model.predict(input_df)[0]

            # Displaying predicted price
            result_label = tk.Label(self.root, text=f"Predicted Price: ${predicted_price:.2f}")
            result_label.pack()
        except Exception as e:
            tk.messagebox.showerror("Error", f"Error predicting price: {str(e)}")

# Calling app
if __name__ == "__main__":
    root = tk.Tk()
    app = HousePricePredictionApp(root)
    root.mainloop()
