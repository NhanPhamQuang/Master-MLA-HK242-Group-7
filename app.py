import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder
import os

# Kiểm tra sự tồn tại của file mô hình
model_file = 'rf.jb'
if not os.path.exists(model_file):
    st.error(f"File mô hình '{model_file}' không tồn tại. Vui lòng kiểm tra thư mục.")
    st.stop()

# Tải mô hình
model = joblib.load(model_file)

# Tạo LabelEncoder thủ công
label_encoders = {}
label_cols = ['Compass', 'Street', 'Ward', 'District', 'Category']

# Danh sách giá trị cho các cột (cần khớp với dữ liệu huấn luyện)
compass_values = ["Đông", "Tây", "Nam", "Bắc", "Đông Bắc", "Đông Nam", "Tây Bắc", "Tây Nam"]
district_values = [
    "Quận 9", "Quận 2", "Bình Thạnh", "Bình Tân", "Quận 7", "Quận 12", "Thủ Đức",
    "Quận 8", "Bình Chánh", "Tân Phú", "Quận 1", "Củ Chi", "Tân Bình", "Gò Vấp",
    "Quận 4", "Nhà Bè", "Quận 3", "Quận 10", "Quận 6", "Phú Nhuận", "Hóc Môn",
    "Quận 5", "Hồ Chí Minh", "Quận 11"
]
category_values = [
    "Căn hộ", "Nhà phố", "Đất nền", "Khác", "Office-tel",
    "Shop-house", "Penthouse", "Biệt thự", "Văn phòng"
]

# Đọc dữ liệu từ file Excel để lấy danh sách Street và Ward
df = pd.read_excel('House_price.xlsx')  # Thay bằng file dữ liệu của bạn
street_values = df['Street'].unique().tolist()
ward_values = df['Ward'].unique().tolist()

# Tạo LabelEncoder
for col, values in zip(
        ['Compass', 'District', 'Category', 'Street', 'Ward'],
        [compass_values, district_values, category_values, street_values, ward_values]
):
    le = LabelEncoder()
    le.fit(values)
    label_encoders[col] = le

# Tiêu đề ứng dụng
st.title("House Price Prediction")
st.write("Enter the Details Below to Predict the House Price")

# Form để nhập dữ liệu
with st.form(key="real_estate_form"):
    # Bedrooms (Số phòng ngủ) - Số nguyên
    bedrooms = st.number_input(
        "Bedrooms:",
        min_value=0,
        max_value=20,
        value=1,
        step=1,
        format="%d"
    )

    # Bathrooms (Số phòng tắm) - Số nguyên
    bathrooms = st.number_input(
        "Bathrooms:",
        min_value=0,
        max_value=20,
        value=1,
        step=1,
        format="%d"
    )

    # Area (Diện tích m²) - Số thực
    area = st.number_input(
        "Area (m²):",
        min_value=0.0,
        max_value=10000.0,
        value=50.0,
        step=0.1,
        format="%.2f"
    )

    # Compass (Hướng nhà) - Hộp chọn
    compass = st.selectbox("Compass:", compass_values)

    # Street (Đường) - Dropdown
    street = st.selectbox("Street:", street_values, index=0)

    # Ward (Phường) - Dropdown
    ward = st.selectbox("Ward:", ward_values, index=0)

    # District (Quận) - Hộp chọn
    district = st.selectbox("District:", district_values)

    # Category (Loại bất động sản) - Hộp chọn
    category = st.selectbox("Category:", category_values)

    # Price per square (Giá mỗi m²) - Giá trị mặc định
    price_per_square = 0.0868590978

    # Balcony (Ban công) - Hộp kiểm
    balcony = st.checkbox("Balcony ?", value=False, help="Tích vào nếu bất động sản có ban công.")
    balcony_value = 1 if balcony else 0

    # Furniture (Nội thất) - Selectbox với 3 giá trị
    furniture = st.selectbox(
        "Furniture:",
        options=[2, 1, 0],
        format_func=lambda x: {2: "Nội thất đầy đủ", 1: "Nội thất cơ bản", 0: "Không có nội thất"}[x],
        help="Chọn mức độ nội thất của bất động sản."
    )
    furniture_value = furniture  # Giá trị trực tiếp từ selectbox

    # Nút gửi
    submit_button = st.form_submit_button(label="Predict")

# Xử lý khi nhấn nút gửi
if submit_button:
    # Tạo dictionary từ dữ liệu nhập
    input_data = {
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Area (m²)': area,
        'Compass': compass,
        'Street': street,
        'Ward': ward,
        'District': district,
        'Category': category,
        'Price per square': price_per_square,
        'Balcony': balcony_value,
        'Furniture': furniture_value
    }

    # Chuyển thành DataFrame
    input_df = pd.DataFrame([input_data])

    # Áp dụng Label Encoding cho các cột categorical
    label_cols = ['Compass', 'Street', 'Ward', 'District', 'Category']
    for col in label_cols:
        try:
            # Kiểm tra xem giá trị có trong classes_ của LabelEncoder không
            if input_df[col].iloc[0] in label_encoders[col].classes_:
                input_df[col] = label_encoders[col].transform(input_df[col])
            else:
                st.error(f"Giá trị '{input_df[col].iloc[0]}' trong cột '{col}' không có trong dữ liệu huấn luyện.")
                st.stop()
        except Exception as e:
            st.error(f"Lỗi khi mã hóa cột '{col}': {str(e)}")
            st.stop()

    # Dự đoán giá nhà
    try:
        predicted_price = model.predict(input_df)[0]
        st.write("### Kết quả dự đoán:")
        st.write(f"**Giá nhà dự đoán**: {predicted_price:.2f} Tỷ VNĐ")
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {str(e)}")
        st.write("Vui lòng kiểm tra định dạng dữ liệu hoặc mô hình.")