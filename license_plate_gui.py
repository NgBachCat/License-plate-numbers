import cv2
import threading
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import pandas as pd
import datetime
import os
import numpy as np
from ultralytics import YOLO
import time
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Global variables
detected_data = []
cap = None
videoPanel = None
resultsPanel = None
imagePanel = None
root = None
stop_camera_flag = False

def update_status(message):
    status_bar.config(text=message)

def on_enter(e):
    e.widget['background'] = '#ffffff'

def on_leave(e, color):
    e.widget['background'] = color

def detect_license_plate(image):
    if image is None:
        return "Lỗi: Không thể tải ảnh.", None

    update_status("Đang xử lý ảnh...")
    results = model(image)
    detections = results[0].boxes

    if detections is not None and len(detections) > 0:
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate = image[y1:y2, x1:x2]

            plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            plate_resized = cv2.resize(plate_gray, (300, 100))

            results = reader.readtext(plate_resized)
            plate_text = ""
            for res in results:
                plate_text += res[-2] + " "
            plate_text = plate_text.strip()
            if plate_text:
                update_status("Đã nhận diện biển số thành công!")
                return plate_text, plate

    update_status("Không tìm thấy biển số xe.")
    return "No license plate detected.", None

def start_camera():
    global cap, stop_camera_flag, videoPanel
    update_status("Đang khởi động camera...")
    stop_camera_flag = False
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không thể kết nối với camera.")
        update_status("Lỗi kết nối camera!")
        return

    if videoPanel is not None:
        videoPanel.destroy()
        videoPanel = None

    threading.Thread(target=update_camera_feed, daemon=True).start()
    update_status("Camera đang hoạt động...")

def stop_camera():
    global stop_camera_flag, cap, videoPanel
    stop_camera_flag = True
    if cap is not None:
        cap.release()
        cap = None
    if videoPanel is not None:
        videoPanel.destroy()
        videoPanel = None
    update_status("Đã dừng camera.")

def update_camera_feed():
    global cap, videoPanel, stop_camera_flag, detected_data, resultsPanel
    
    while True:
        if stop_camera_flag:
            break

        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame_rgb)
            frame_image = ImageTk.PhotoImage(frame_image)

            if videoPanel is None:
                videoPanel = Label(display_frame, image=frame_image)
                videoPanel.image = frame_image
                videoPanel.pack(expand=True, fill=BOTH)
            else:
                videoPanel.configure(image=frame_image)
                videoPanel.image = frame_image

            if time.time() % 0.5 < 0.1:
                plate_text, plate_image = detect_license_plate(frame)
                if plate_image is not None and plate_text != "No license plate detected.":
                    province, number = extract_province_and_number(plate_text)
                    detected_data.append({
                        "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "Date": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "Plate": plate_text,
                        "Province": province,
                        "Number": number
                    })
                    update_results_display(plate_text, province, number)

            root.update()
            time.sleep(0.01)

def update_results_display(plate_text, province, number):
    global resultsPanel

    if resultsPanel is not None:
        resultsPanel.destroy()  # Xóa giao diện cũ nếu có

    # Tạo frame mới cho từng mục thông tin
    resultsPanel = Frame(info_frame, bg="white", relief="ridge", borderwidth=2)
    resultsPanel.pack(fill=X, padx=5, pady=5)

    # Biển số xe
    plate_label = Label(resultsPanel, text="Biển số xe:", font=("Helvetica", 12, "bold"), bg="white", anchor="w")
    plate_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
    plate_value = Label(resultsPanel, text=plate_text, font=("Helvetica", 12), bg="white", fg="#333", anchor="w")
    plate_value.grid(row=0, column=1, sticky="w", padx=5, pady=5)

    # Tỉnh/Thành phố
    province_label = Label(resultsPanel, text="Tỉnh/TP:", font=("Helvetica", 12, "bold"), bg="white", anchor="w")
    province_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
    province_value = Label(resultsPanel, text=province, font=("Helvetica", 12), bg="white", fg="#333", anchor="w")
    province_value.grid(row=1, column=1, sticky="w", padx=5, pady=5)

    # Số xe
    number_label = Label(resultsPanel, text="Số xe:", font=("Helvetica", 12, "bold"), bg="white", anchor="w")
    number_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
    number_value = Label(resultsPanel, text=number, font=("Helvetica", 12), bg="white", fg="#333", anchor="w")
    number_value.grid(row=2, column=1, sticky="w", padx=5, pady=5)

    # Căn chỉnh cột để các thông tin đều nhau
    resultsPanel.columnconfigure(0, weight=1)
    resultsPanel.columnconfigure(1, weight=3)

def extract_province_and_number(plate_text):
    province_codes = {
        "41": "TP. Hồ Chí Minh", "50": "TP. Hồ Chí Minh", "51": "TP. Hồ Chí Minh",
        "52": "TP. Hồ Chí Minh", "53": "TP. Hồ Chí Minh", "54": "TP. Hồ Chí Minh",
        "55": "TP. Hồ Chí Minh", "56": "TP. Hồ Chí Minh", "57": "TP. Hồ Chí Minh",
        "58": "TP. Hồ Chí Minh", "59": "TP. Hồ Chí Minh",
        "29": "Hà Nội", "30": "Hà Nội", "31": "Hà Nội", "32": "Hà Nội",
        "33": "Hà Nội", "40": "Hà Nội",
        "43": "Đà Nẵng",
        "61": "Bình Dương",
        "39": "Đồng Nai", "60": "Đồng Nai",
        "79": "Khánh Hòa",
        "15": "Hải Phòng", "16": "Hải Phòng",
        "62": "Long An",
        "92": "Quảng Nam",
        "72": "Bà Rịa - Vũng Tàu",
        "47": "Đắk Lắk",
        "65": "Cần Thơ",
        "86": "Bình Thuận",
        "49": "Lâm Đồng",
        "75": "Thừa Thiên Huế",
        "68": "Kiên Giang",
        "99": "Bắc Ninh",
        "14": "Quảng Ninh",
        "36": "Thanh Hóa",
        "37": "Nghệ An",
        "34": "Hải Dương",
        "81": "Gia Lai",
        "93": "Bình Phước",
        "89": "Hưng Yên",
        "77": "Bình Định",
        "63": "Tiền Giang",
        "17": "Thái Bình",
        "98": "Bắc Giang",
        "28": "Hòa Bình",
        "67": "An Giang",
        "88": "Vĩnh Phúc",
        "70": "Tây Ninh",
        "20": "Thái Nguyên",
        "24": "Lào Cai",
        "18": "Nam Định",
        "76": "Quảng Ngãi",
        "71": "Bến Tre",
        "48": "Đắk Nông",
        "69": "Cà Mau",
        "64": "Vĩnh Long",
        "35": "Ninh Bình",
        "19": "Phú Thọ",
        "85": "Ninh Thuận",
        "78": "Phú Yên",
        "90": "Hà Nam",
        "38": "Hà Tĩnh",
        "66": "Đồng Tháp",
        "83": "Sóc Trăng",
        "82": "Kon Tum",
        "73": "Quảng Bình",
        "74": "Quảng Trị",
        "84": "Trà Vinh",
        "95": "Hậu Giang",
        "26": "Sơn La",
        "94": "Bạc Liêu",
        "21": "Yên Bái",
        "22": "Tuyên Quang",
        "27": "Điện Biên",
        "25": "Lai Châu",
        "12": "Lạng Sơn",
        "23": "Hà Giang",
        "97": "Bắc Kạn",
        "11": "Cao Bằng"
    }
    if len(plate_text) >= 2:
        province_code = plate_text[:2]
        province = province_codes.get(province_code, "Không xác định")
        number = plate_text[2:]
        return province, number
    return "Không xác định", plate_text

def select_image():
    global resultsPanel, imagePanel
    update_status("Đang chọn ảnh...")
    path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])
    
    if path:
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Lỗi", "Không thể đọc ảnh. Vui lòng thử lại.")
            return
            
        if imagePanel is not None:
            imagePanel.destroy()
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_pil = image_pil.resize((640, 480), Image.Resampling.LANCZOS)
        image_tk = ImageTk.PhotoImage(image_pil)

        imagePanel = Label(display_frame, image=image_tk)
        imagePanel.image = image_tk
        imagePanel.pack(expand=True, fill=BOTH)

        plate_text, plate_image = detect_license_plate(image)
        if plate_text != "No license plate detected.":
            province, number = extract_province_and_number(plate_text)
            update_results_display(plate_text, province, number)

def export_to_excel():
    global detected_data
    if not detected_data:
        messagebox.showwarning("Cảnh báo", "Chưa có dữ liệu để xuất.")
        update_status("Không có dữ liệu để xuất!")
        return

    try:
        df = pd.DataFrame(detected_data)
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile="detected_license_plates.xlsx"
        )
        if file_path:
            df.to_excel(file_path, index=False, engine='openpyxl')  # Đảm bảo sử dụng engine openpyxl
            messagebox.showinfo("Thành công", f"Đã xuất dữ liệu thành công vào file {file_path}")
            update_status("Xuất dữ liệu thành công!")
    except ImportError:
        messagebox.showerror("Lỗi", "Vui lòng cài đặt thư viện openpyxl bằng lệnh: pip install openpyxl")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Lỗi khi xuất dữ liệu: {e}")
        update_status("Lỗi xuất dữ liệu!")

# GUI Setup
root = Tk()
root.title("Hệ Thống Nhận Diện Biển Số Xe")
root.geometry('1200x800')
root.configure(bg="#f0f0f0")

# Style configuration
style = ttk.Style()
style.configure('TFrame', background='#f0f0f0')
style.configure('TButton', padding=6, relief="flat", background="#2980b9")

# Main title
title_label = Label(root, 
                   text="HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE",
                   font=("Helvetica", 24, "bold"),
                   bg="#f0f0f0",
                   fg="#2C3E50",
                   pady=20)
title_label.pack(side=TOP, fill=X)

# Create main container
main_container = Frame(root, bg="#f0f0f0")
main_container.pack(expand=True, fill=BOTH, padx=20, pady=10)

# Button frame (left side)
button_frame = Frame(main_container, bg="#f0f0f0", relief="ridge", borderwidth=2)
button_frame.pack(side=LEFT, fill=Y, padx=10, pady=5)

# Display frame (center)
display_frame = Frame(main_container, bg="white", relief="ridge", borderwidth=2)
display_frame.pack(side=LEFT, expand=True, fill=BOTH, padx=10, pady=5)

# Information frame (right side)
info_frame = Frame(main_container, bg="white", relief="ridge", borderwidth=2, width=200)
info_frame.pack(side=LEFT, fill=Y, padx=10, pady=5)

# Button styles
button_style = {
    'font': ("Helvetica", 12),
    'width': 15,
    'height': 2,
    'relief': 'raised',
    'cursor': 'hand2'
}

button_colors = {
    'camera': '#3498DB',
    'image': '#2ECC71',
    'export': '#E74C3C',
    'stop': '#E67E22',
    'exit': '#95A5A6'
}

# Create buttons
btnCamera = Button(button_frame, text="Bắt đầu camera", command=start_camera,
                  bg=button_colors['camera'], fg='white', **button_style)
btnCamera.pack(pady=10, padx=5)

btnStopCamera = Button(button_frame, text="Dừng camera", command=stop_camera,
                      bg=button_colors['stop'], fg='white', **button_style)
btnStopCamera.pack(pady=10, padx=5)

btnImage = Button(button_frame, text="Chọn ảnh", command=select_image,
                 bg=button_colors['image'], fg='white', **button_style)
btnImage.pack(pady=10, padx=5)

btnExport = Button(button_frame, text="Xuất Excel", command=export_to_excel,
                  bg=button_colors['export'], fg='white', **button_style)
btnExport.pack(pady=10, padx=5)

btnExit = Button(button_frame, text="Thoát", command=root.quit,
                bg=button_colors['exit'], fg='white', **button_style)
btnExit.pack(pady=10, padx=5)

# Add hover effects to buttons
for btn in [btnCamera, btnImage, btnExport, btnStopCamera, btnExit]:
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", lambda e, color=btn['background']: on_leave(e, color))

# Status bar
status_bar = Label(root, text="Sẵn sàng...", bd=1, relief=SUNKEN, anchor=W)
status_bar.pack(side=BOTTOM, fill=X)

# Information label in info_frame
info_label = Label(info_frame, text="THÔNG TIN BIỂN SỐ",
                  font=("Helvetica", 14, "bold"),
                  bg="white", fg="#2C3E50")
info_label.pack(pady=10)

root.mainloop()
