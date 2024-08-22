### 1. **Khởi tạo Dữ liệu & Tiền xử lý**
**Code cũ:**
```python
import pandas as pd

data = pd.read_csv('dataset.csv')
```
- **Nhược điểm:** Mã này chỉ đơn giản đọc dữ liệu từ file CSV mà không có xử lý dữ liệu trước khi sử dụng. Nó cũng không rõ ràng về việc dữ liệu được sử dụng ở đâu và như thế nào.

**Code mới:**
```python
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    # Loại bỏ cột không cần thiết
    data = data.drop(columns=['UnnecessaryColumn'])
    data.fillna(0, inplace=True)
    return data

data = load_data('dataset.csv')
if data is not None:
    data = preprocess_data(data)
```
- **Ưu điểm:** Mã mới phân chia rõ ràng thành các hàm `load_data` và `preprocess_data`. Việc sử dụng `try-except` để xử lý lỗi khi load dữ liệu giúp mã an toàn hơn. Hàm `preprocess_data` giúp loại bỏ cột không cần thiết và xử lý giá trị thiếu, giúp chuẩn bị dữ liệu tốt hơn.

### 2. **Mã hóa dữ liệu**
**Code cũ:**
```python
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
data['Category'] = labelencoder.fit_transform(data['Category'])
```
- **Nhược điểm:** Mã này chỉ mã hóa một cột duy nhất mà không rõ cách xử lý những cột khác. Điều này không linh hoạt nếu cần mã hóa nhiều cột hoặc cần sử dụng nhiều kỹ thuật mã hóa khác nhau.

**Code mới:**
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.sparse import hstack

def encode_data(data):
    labelencoder = LabelEncoder()
    data['Category'] = labelencoder.fit_transform(data['Category'])
    
    # Mã hóa OneHot cho nhiều cột
    onehotencoder = OneHotEncoder()
    X_onehot = onehotencoder.fit_transform(data[['CategoricalColumn1', 'CategoricalColumn2']])
    
    # Kết hợp dữ liệu mã hóa
    X_combined = hstack([data[['NumericColumn']], X_onehot])
    
    return X_combined

encoded_data = encode_data(data)
```
- **Ưu điểm:** Code mới linh hoạt hơn với khả năng mã hóa nhiều cột cùng một lúc (LabelEncoding cho `Category` và OneHotEncoding cho các cột `CategoricalColumn1`, `CategoricalColumn2`). Việc kết hợp các dữ liệu mã hóa bằng `hstack` giúp tối ưu hóa hiệu quả xử lý dữ liệu.

### 3. **Huấn luyện Mô hình**
**Code cũ:**
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
```
- **Nhược điểm:** Chỉ sử dụng một mô hình duy nhất mà không có khả năng so sánh với các mô hình khác. Điều này làm giảm tính linh hoạt và khả năng tối ưu hóa kết quả.

**Code mới:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    models = {
        "RandomForest": RandomForestClassifier(),
        "LogisticRegression": LogisticRegression()
    }
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

models = train_model(X_train, y_train)
```
- **Ưu điểm:** Mã mới có khả năng huấn luyện nhiều mô hình (RandomForest và LogisticRegression) trong cùng một lần chạy. Điều này giúp dễ dàng so sánh các mô hình khác nhau và chọn mô hình phù hợp nhất.

### 4. **Đánh giá Mô hình**
**Code cũ:**
```python
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```
- **Nhược điểm:** Chỉ đánh giá một mô hình với một chỉ số đơn giản (accuracy). Điều này có thể không đủ để đánh giá chính xác hiệu suất của mô hình, đặc biệt là với dữ liệu không cân bằng.

**Code mới:**
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"Model: {name}")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("-" * 30)

evaluate_model(models, X_test, y_test)
```
- **Ưu điểm:** Mã mới đánh giá các mô hình với nhiều chỉ số khác nhau như `accuracy_score`, `confusion_matrix`, và `classification_report`. Điều này giúp cung cấp cái nhìn toàn diện hơn về hiệu suất của mỗi mô hình.

### 5. **Giao diện người dùng**
**Code cũ:** Không có giao diện người dùng.

**Code mới:**
```python
import tkinter as tk
from tkinter import messagebox

def show_result(model, X_test):
    y_pred = model.predict(X_test)
    messagebox.showinfo("Prediction", f"Predicted class: {y_pred[0]}")

def create_ui(models, X_test):
    root = tk.Tk()
    root.title("Model Prediction")

    model_choice = tk.StringVar()
    model_choice.set("RandomForest")

    for name in models.keys():
        tk.Radiobutton(root, text=name, variable=model_choice, value=name).pack()

    tk.Button(root, text="Predict", command=lambda: show_result(models[model_choice.get()], X_test)).pack()
    
    root.mainloop()

create_ui(models, X_test)
```
- **Ưu điểm:** Mã mới cung cấp một giao diện người dùng cơ bản với Tkinter, cho phép người dùng lựa chọn mô hình và xem kết quả dự đoán. Điều này giúp người dùng cuối dễ dàng tương tác mà không cần phải chỉnh sửa mã nguồn.

### **Kết luận tổng quát:**
- **Code cũ:** Thích hợp cho các ứng dụng nhỏ và đơn giản, nhưng thiếu tính linh hoạt và khó mở rộng, bảo trì.
- **Code mới:** Cung cấp một cấu trúc rõ ràng, dễ bảo trì, mở rộng và hiệu quả hơn. Nó phù hợp với các dự án lớn hoặc cần sự linh hoạt trong việc lựa chọn mô hình và xử lý dữ liệu.
