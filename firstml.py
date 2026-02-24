import streamlit as st
import numpy as np

# --- 1. البيانات والتدريب (تلقائي عند تشغيل التطبيق) ---
x_train = np.array([
    [3.0, 0.0, 302.0, 0.0],  # مثال الشقة التي أرفقتها
    [3.0, 5.0, 150.0, 2.0],  
    [2.0, 10.0, 100.0, 1.0], 
    [4.0, 1.0, 200.0, 3.0],  
    [3.0, 0.5, 120.0, 0.0]
])
y_train = np.array([88.7, 45.0, 28.0, 65.0, 40.0])

x_mean = np.mean(x_train, axis=0)
x_std = np.std(x_train, axis=0)
x_scaled = (x_train - x_mean) / x_std

def train_model(x, y):
    m, n = x.shape
    w = np.zeros(n)
    b = 0.0
    for _ in range(2000):
        err = (np.dot(x, w) + b) - y
        w -= 0.01 * (np.dot(x.T, err) / m)
        b -= 0.01 * (np.sum(err) / m)
    return w, b

w, b = train_model(x_scaled, y_train)

# --- 2. واجهة التطبيق (Streamlit) ---
st.set_page_config(page_title="مخمن أسعار العقارات", page_icon="🏠")
st.title("🏠 نظام التنبؤ بأسعار الشقق")
st.write("أدخل مواصفات الشقة للحصول على السعر التقديري")

col1, col2 = st.columns(2)

with col1:
    rooms = st.number_input("عدد الغرف", min_value=1.0, max_value=10.0, value=3.0)
    age = st.number_input("عمر البناء (بالسنوات)", min_value=0.0, value=0.0)

with col2:
    area = st.number_input("المساحة (متر مربع)", min_value=50.0, value=300.0)
    floor = st.number_input("الطابق", min_value=0.0, value=0.0)

if st.button("احسب السعر المتوقع"):
    # معالجة المدخلات
    x_input = np.array([rooms, age, area, floor])
    x_input_scaled = (x_input - x_mean) / x_std
    
    # التنبؤ
    prediction = np.dot(x_input_scaled, w) + b
    final_price = max(0, prediction)
    
    st.success(f"💰 السعر المتوقع: {final_price:,.0f} دينار أردني")
    st.info("ملاحظة: هذا السعر تقديري بناءً على البيانات المتوفرة.")