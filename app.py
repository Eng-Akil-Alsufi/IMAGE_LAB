from flask import Flask, render_template, request, redirect, url_for, jsonify
from PIL import Image, ImageEnhance
import os
import uuid
import numpy as np
from scipy.fftpack import dct
import cv2
import exifread
from skimage import feature, filters
import pywt  # لتحويل المويجات
from sklearn import svm  # لدعم المتجهات
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import colorsys
from collections import Counter
import webcolors

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def index():
    return render_template('p1.html')

@app.route('/p1')
def show_p1():
    return render_template('p1.html')

@app.route('/p2')
def show_p2():
    return render_template('p2.html')

@app.route('/p3')
def page3():
    return render_template('p3.html')

@app.route('/p4')
def page4():
    return render_template('p4.html')

@app.route('/p5')
def page5():
    return render_template('p5.html')

@app.route('/result2')
def show_result2_page():
    # استخراج المعاملات بشكل أكثر تفصيلاً
    original_image = request.args.get('filename')
    manipulation_result = request.args.get('result')
    
    # طباعة المعاملات للتديج والتأكد
    print("Debug - show_result2_page:")
    print(f"Original Image: {original_image}")
    print(f"Manipulation Result: {manipulation_result}")
    
    # التحقق المفصل من القيم
    if not original_image:
        print("Error: No filename provided")
        return "خطأ: لم يتم تمرير اسم الصورة", 400
    
    if not manipulation_result:
        print("Error: No manipulation result provided")
        return "خطأ: لم يتم تمرير نتيجة الكشف عن التلاعب", 400
    
    # التحقق من وجود الصورة
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], original_image)
    if not os.path.exists(image_path):
        print(f"Error: Image not found - {image_path}")
        return f"خطأ: الصورة {original_image} غير موجودة", 404
    
    # عرض القالب مع التأكد من وجود البيانات
    try:
        return render_template('result2.html', 
                               filename=original_image, 
                               result=manipulation_result)
    except Exception as e:
        print(f"Error rendering template: {e}")
        return "خطأ في عرض النتائج", 500


@app.route('/detect_manipulation', methods=['POST'])
def detect_manipulation():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
    
    if file:
        # توليد اسم ملف فريد
        filename = f"upload_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # استدعاء دالة الكشف عن التلاعب
            manipulation_result = detect_image_manipulation(filepath)

            return jsonify({
                'success': True, 
                'filename': filename, 
                'result': manipulation_result['message'],
                'redirect': url_for('show_result2_page', filename=filename, result=manipulation_result['message'])
            })
        except Exception as e:
            print(f"Error detecting manipulation: {e}")
            return jsonify({
                'success': False, 
                'message': f'حدث خطأ أثناء تحليل الصورة: {str(e)}'
            })

def detect_image_manipulation(image_path):
    try:
        # قراءة الصورة
        img = cv2.imread(image_path)
        
        # التحقق من قراءة الصورة بشكل صحيح
        if img is None:
            return {'message': 'خطأ في قراءة الصورة'}
        
        # تحويل الصورة إلى تدرج الرمادي
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # تطبيق DCT
        dct_result = dct(dct(gray.T, norm='ortho').T, norm='ortho')
        
        # حساب متوسط القيم المطلقة للترددات العالية
        high_freq_mean = np.mean(np.abs(dct_result[int(dct_result.shape[0]/2):, int(dct_result.shape[1]/2):]))
        
        # تحليل الخصائص الإحصائية
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # تحليل الحواف
        edges = cv2.Canny(gray, 100, 200)
        edge_count = np.sum(edges > 0)
        
        # تحليل التباين اللوني
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color_variance = np.var(hsv_image[:, :, 1])  # تباين اللون
        
        # تحليل البيانات الوصفية
        camera_model = 'غير معروف'
        date_time = 'غير معروف'
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f)
                camera_model = str(tags.get('Image Model', 'غير معروف'))
                date_time = str(tags.get('EXIF DateTimeOriginal', 'غير معروف'))
        except Exception as e:
            print(f"خطأ في استخراج البيانات الوصفية: {e}")
        
        # تحديد عتبات للكشف عن التلاعب
        threshold_dct = 5
        threshold_edge_count = 1000
        threshold_color_variance = 100
        
        # الكشف عن التلاعب بناءً على التحليلات
        if (high_freq_mean > threshold_dct and 
            std_intensity > 20 and 
            edge_count > threshold_edge_count and 
            color_variance > threshold_color_variance):
            return {
                'message': f"تم الكشف عن تلاعب محتمل في الصورة. الكاميرا: {camera_model}, تاريخ الالتقاط: {date_time}"
            }
        else:
            return {
                'message': "لم يتم الكشف عن تلاعب واضح في الصورة"
            }
    except Exception as e:
        print(f"خطأ في تحليل الصورة: {e}")
        return {
            'message': f'حدث خطأ أثناء تحليل الصورة: {str(e)}'
        }

@app.route('/uploads', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
    
    if file:
        # توليد اسم ملف فريد
        filename = f"upload_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # حفظ الصورة
            file.save(filepath)
            
            return jsonify({
                'success': True, 
                'filename': filename,
                'message': 'تم رفع الصورة بنجاح'
            })
        except Exception as e:
            print(f"Error saving file: {e}")
            return jsonify({
                'success': False, 
                'message': f'حدث خطأ أثناء حفظ الصورة: {str(e)}'
            })

def get_color_name(rgb):
    color_dict = {
        (255, 0, 0): 'أحمر',
        (0, 255, 0): 'أخضر',
        (0, 0, 255): 'أزرق',
        (255, 255, 0): 'أصفر',
        (255, 0, 255): 'أرجواني',
        (0, 255, 255): 'سماوي',
        (0, 0, 0): 'أسود',
        (255, 255, 255): 'أبيض',
        (128, 128, 128): 'رمادي'
    }
    
    # إيجاد أقرب لون
    min_distance = float('inf')
    closest_color_name = 'اللون غير محدد'
    
    try:
        for color, name in color_dict.items():
            # استخدام np.sum بدلاً من sum
            distance = np.sum(np.square(np.array(rgb) - np.array(color)))
            if distance < min_distance:
                min_distance = distance
                closest_color_name = name
    except Exception as e:
        print(f"Error in get_color_name: {e}")
    
    return closest_color_name

def convert_to_serializable(obj):
    """
    تحويل الكائنات غير القابلة للتسلسل إلى أنواع قابلة للتسلسل
    """
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    elif isinstance(obj, np.uint8):
        return int(obj)
    else:
        return obj
def analyze_image_colors(image_path):
    """
    تحليل الألوان البارزة في الصورة
    :param image_path: مسار الصورة
    :return: قائمة بالألوان الرئيسية وتوزيعها
    """
    # فتح الصورة
    image = Image.open(image_path)
    
    # تحويل الصورة إلى وضع RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # تقليل حجم الصورة للتحليل السريع
    image = image.resize((150, 150))
    
    # تحويل الصورة إلى مصفوفة
    img_array = np.array(image)
    
    # إعادة تشكيل المصفوفة
    pixels = img_array.reshape(-1, 3)
    
    # حساب الألوان الفريدة وعددها
    color_counts = Counter(map(tuple, pixels))
    
    # استخراج أهم 10 ألوان
    top_colors = color_counts.most_common(10)
    
    # حساب العدد الإجمالي للبكسلات
    total_pixels = sum(count for _, count in top_colors)
    
    # تحليل الألوان
    color_analysis = []
    
    for color, count in top_colors:
        # حساب النسبة المئوية
        percentage = (count / total_pixels) * 100
        
        # تحويل RGB إلى HEX
        hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
        
        # تحديد اسم اللون
        color_name = get_color_name(color)
        
        # تحويل RGB إلى HSV للتحليل
        hsv_color = colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)
        
        color_analysis.append({
            'hex': hex_color,
            'rgb': list(map(int, color)),  # تأكد من تحويل القيم إلى أعداد صحيحة
            'name': color_name,
            'percentage': round(percentage, 2),
            'hue': round(hsv_color[0] * 360, 2),
            'saturation': round(hsv_color[1] * 100, 2),
            'value': round(hsv_color[2] * 100, 2)
        })
    
    return color_analysis

@app.route('/improve', methods=['POST'])
def improve_image():
    print("Received improve request")
    
    # التحقق من وجود الصورة
    if 'image' not in request.files and 'filename' not in request.form:
        print("No image or filename provided")
        return jsonify({'success': False, 'message': 'لم يتم تقديم صورة'})

    try:
        # معالجة رفع الصورة الجديدة
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'success': False, 'message': 'لم يتم اختيار صورة'})
            
            # توليد اسم ملف فريد
            filename = f"upload_{uuid.uuid4().hex[:8]}.png"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        else:
            # استخدام الصورة الموجودة مسبقًا
            filename = request.form.get('filename')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            if not os.path.exists(filepath):
                print("File not found:", filepath)
                return jsonify({'success': False, 'message': 'الملف غير موجود'})

        # استلام قيم أشرطة التمرير
        brightness_value = float(request.form.get('brightness', 20)) / 100 + 1
        contrast_value = float(request.form.get('contrast', 50)) / 100 + 1
        sharpness_value = float(request.form.get('sharpness', 50)) / 100 + 1
        color_value = float(request.form.get('color', 30)) / 100 + 1
        resolution_value = float(request.form.get('resolution', 30)) / 100 + 1
        print("Processing values:", {
            'brightness': brightness_value,
            'contrast': contrast_value,
            'sharpness': sharpness_value,
            'color': color_value,
            'resolution': resolution_value,

        })

        print("Opening image")
        image = Image.open(filepath)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            
        print("Applying enhancements")
        # تحسين الصورة
        enhanced_image = enhance_brightness(image, brightness_value)
        enhanced_image = enhance_contrast(enhanced_image, contrast_value)
        enhanced_image = enhance_sharpness(enhanced_image, sharpness_value)
        enhanced_image = enhance_color(enhanced_image, color_value)
        enhanced_image = enhance_resolution(enhanced_image, resolution_value)

        enhanced_filename = f"enhanced_{filename}"
        enhanced_filepath = os.path.join(app.config['UPLOAD_FOLDER'], enhanced_filename)
        print("Saving enhanced image to:", enhanced_filepath)
        enhanced_image.save(enhanced_filepath)
        
        
        # تحليل الألوان
        color_analysis = analyze_image_colors(filepath)

        # تحويل البيانات للتأكد من إمكانية التسلسل
        serializable_color_analysis = [
            {k: convert_to_serializable(v) for k, v in color.items()} 
            for color in color_analysis
        ]

        return jsonify({
            'success': True,
            'original': filename,
            'enhanced': enhanced_filename,
            'color_analysis': serializable_color_analysis,
            'redirect': url_for('show_result', original=filename, enhanced=enhanced_filename)
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'success': False, 'message': str(e)})
# إضافة route جديد لعرض نتائج الألوان
@app.route('/color_result', methods=['GET', 'POST'])
def color_result():
    if request.method == 'POST':
        # استقبال بيانات تحليل الألوان من الصفحة السابقة
        original_image = request.form.get('original_image')
        color_analysis_str = request.form.get('color_analysis')
        
        # تحويل السلسلة إلى قائمة
        try:
            color_analysis = eval(color_analysis_str)
        except Exception as e:
            print(f"Error parsing color analysis: {e}")
            color_analysis = []
        
        return render_template('result3.html', 
                               original_image=original_image, 
                               color_analysis=color_analysis)
    
    # للطلبات GET
    original_image = request.args.get('original_image')
    color_analysis_str = request.args.get('color_analysis')
    
    try:
        color_analysis = eval(color_analysis_str) if color_analysis_str else []
    except Exception as e:
        print(f"Error parsing color analysis: {e}")
        color_analysis = []
    
    return render_template('result3.html', 
                       original_image=original_image, 
                       color_analysis=color_analysis)
# لعرض نتائج التحسين على صفحة result
@app.route('/result')
def show_result():
    original_image = request.args.get('original')
    enhanced_image = request.args.get('enhanced')
    
    # التحقق من وجود القيم
    if not original_image or not enhanced_image:
        print("Error: Missing original or enhanced image parameters")
        return "خطأ: معلومات الصورة غير مكتملة", 400
    
    # التحقق من أن القيم هي نصوص
    if not isinstance(original_image, str) or not isinstance(enhanced_image, str):
        print("Error: Image parameters must be strings")
        return "خطأ: يجب أن تكون معلومات الصورة نصية", 400
    
    # التحقق من وجود الصور
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_image)
    enhanced_path = os.path.join(app.config['UPLOAD_FOLDER'], enhanced_image)
    
    # طباعة المسارات للتأكد
    print(f"Original path: {original_path}")
    print(f"Enhanced path: {enhanced_path}")
    
    if not os.path.exists(original_path):
        print(f"Original image not found: {original_path}")
        return f"خطأ: الصورة الأصلية {original_image} غير موجودة", 404
    
    if not os.path.exists(enhanced_path):
        print(f"Enhanced image not found: {enhanced_path}")
        return f"خطأ: الصورة المحسنة {enhanced_image} غير موجودة", 404
    
    return render_template('result.html', 
                           original_image=original_image, 
                           enhanced_image=enhanced_image)

# دوال تحسين الصورة المحدثة مع معاملات مخصصة
def enhance_brightness(image, factor):
    """
    تحسين سطوع الصورة بمعامل مخصص
    :param image: الصورة الأصلية
    :param factor: معامل السطوع (1.0 = بدون تغيير)
    """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def enhance_contrast(image, factor):
    """
    تحسين تباين الصورة بمعامل مخصص
    :param image: الصورة الأصلية
    :param factor: معامل التباين (1.0 = بدون تغيير)
    """
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def enhance_sharpness(image, factor):
    """
    تحسين حدة الصورة بمعامل مخصص
    :param image: الصورة الأصلية
    :param factor: معامل الحدة (1.0 = بدون تغيير)
    """
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)

def enhance_color(image, factor):
    """
    تحسين تشبع الألوان بمعامل مخصص
    :param image: الصورة الأصلية
    :param factor: معامل تشبع الألوان (1.0 = بدون تغيير)
    """
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)


def enhance_resolution(image, scale_factor):
    """
    تحسين دقة الصورة
    :param image: الصورة الأصلية
    :param scale_factor: معامل تغيير الحجم
    """
    width, height = image.size
    new_size = (int(width * scale_factor), int(height * scale_factor))
    return image.resize(new_size, Image.LANCZOS)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
from flask import Flask, render_template, request, redirect, url_for, jsonify
from PIL import Image, ImageEnhance
import os
import uuid
import numpy as np
from scipy.fftpack import dct
import cv2
import exifread
from skimage import feature, filters
import pywt  # لتحويل المويجات
from sklearn import svm  # لدعم المتجهات
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import colorsys
from collections import Counter
import webcolors

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def index():
    return render_template('p1.html')

@app.route('/p1')
def show_p1():
    return render_template('p1.html')

@app.route('/p2')
def show_p2():
    return render_template('p2.html')

@app.route('/p3')
def page3():
    return render_template('p3.html')

@app.route('/p4')
def page4():
    return render_template('p4.html')

@app.route('/p5')
def page5():
    return render_template('p5.html')

@app.route('/result2')
def show_result2_page():
    # استخراج المعاملات بشكل أكثر تفصيلاً
    original_image = request.args.get('filename')
    manipulation_result = request.args.get('result')
    
    # طباعة المعاملات للتديج والتأكد
    print("Debug - show_result2_page:")
    print(f"Original Image: {original_image}")
    print(f"Manipulation Result: {manipulation_result}")
    
    # التحقق المفصل من القيم
    if not original_image:
        print("Error: No filename provided")
        return "خطأ: لم يتم تمرير اسم الصورة", 400
    
    if not manipulation_result:
        print("Error: No manipulation result provided")
        return "خطأ: لم يتم تمرير نتيجة الكشف عن التلاعب", 400
    
    # التحقق من وجود الصورة
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], original_image)
    if not os.path.exists(image_path):
        print(f"Error: Image not found - {image_path}")
        return f"خطأ: الصورة {original_image} غير موجودة", 404
    
    # عرض القالب مع التأكد من وجود البيانات
    try:
        return render_template('result2.html', 
                               filename=original_image, 
                               result=manipulation_result)
    except Exception as e:
        print(f"Error rendering template: {e}")
        return "خطأ في عرض النتائج", 500


@app.route('/detect_manipulation', methods=['POST'])
def detect_manipulation():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
    
    if file:
        # توليد اسم ملف فريد
        filename = f"upload_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # استدعاء دالة الكشف عن التلاعب
            manipulation_result = detect_image_manipulation(filepath)

            return jsonify({
                'success': True, 
                'filename': filename, 
                'result': manipulation_result['message'],
                'redirect': url_for('show_result2_page', filename=filename, result=manipulation_result['message'])
            })
        except Exception as e:
            print(f"Error detecting manipulation: {e}")
            return jsonify({
                'success': False, 
                'message': f'حدث خطأ أثناء تحليل الصورة: {str(e)}'
            })

def detect_image_manipulation(image_path):
    try:
        # قراءة الصورة
        img = cv2.imread(image_path)
        
        # التحقق من قراءة الصورة بشكل صحيح
        if img is None:
            return {'message': 'خطأ في قراءة الصورة'}
        
        # تحويل الصورة إلى تدرج الرمادي
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # تطبيق DCT
        dct_result = dct(dct(gray.T, norm='ortho').T, norm='ortho')
        
        # حساب متوسط القيم المطلقة للترددات العالية
        high_freq_mean = np.mean(np.abs(dct_result[int(dct_result.shape[0]/2):, int(dct_result.shape[1]/2):]))
        
        # تحليل الخصائص الإحصائية
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # تحليل الحواف
        edges = cv2.Canny(gray, 100, 200)
        edge_count = np.sum(edges > 0)
        
        # تحليل التباين اللوني
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color_variance = np.var(hsv_image[:, :, 1])  # تباين اللون
        
        # تحليل البيانات الوصفية
        camera_model = 'غير معروف'
        date_time = 'غير معروف'
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f)
                camera_model = str(tags.get('Image Model', 'غير معروف'))
                date_time = str(tags.get('EXIF DateTimeOriginal', 'غير معروف'))
        except Exception as e:
            print(f"خطأ في استخراج البيانات الوصفية: {e}")
        
        # تحديد عتبات للكشف عن التلاعب
        threshold_dct = 5
        threshold_edge_count = 1000
        threshold_color_variance = 100
        
        # الكشف عن التلاعب بناءً على التحليلات
        if (high_freq_mean > threshold_dct and 
            std_intensity > 20 and 
            edge_count > threshold_edge_count and 
            color_variance > threshold_color_variance):
            return {
                'message': f"تم الكشف عن تلاعب محتمل في الصورة. الكاميرا: {camera_model}, تاريخ الالتقاط: {date_time}"
            }
        else:
            return {
                'message': "لم يتم الكشف عن تلاعب واضح في الصورة"
            }
    except Exception as e:
        print(f"خطأ في تحليل الصورة: {e}")
        return {
            'message': f'حدث خطأ أثناء تحليل الصورة: {str(e)}'
        }

@app.route('/uploads', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
    
    if file:
        # توليد اسم ملف فريد
        filename = f"upload_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # حفظ الصورة
            file.save(filepath)
            
            return jsonify({
                'success': True, 
                'filename': filename,
                'message': 'تم رفع الصورة بنجاح'
            })
        except Exception as e:
            print(f"Error saving file: {e}")
            return jsonify({
                'success': False, 
                'message': f'حدث خطأ أثناء حفظ الصورة: {str(e)}'
            })

def get_color_name(rgb):
    color_dict = {
        (255, 0, 0): 'أحمر',
        (0, 255, 0): 'أخضر',
        (0, 0, 255): 'أزرق',
        (255, 255, 0): 'أصفر',
        (255, 0, 255): 'أرجواني',
        (0, 255, 255): 'سماوي',
        (0, 0, 0): 'أسود',
        (255, 255, 255): 'أبيض',
        (128, 128, 128): 'رمادي'
    }
    
    # إيجاد أقرب لون
    min_distance = float('inf')
    closest_color_name = 'اللون غير محدد'
    
    try:
        for color, name in color_dict.items():
            # استخدام np.sum بدلاً من sum
            distance = np.sum(np.square(np.array(rgb) - np.array(color)))
            if distance < min_distance:
                min_distance = distance
                closest_color_name = name
    except Exception as e:
        print(f"Error in get_color_name: {e}")
    
    return closest_color_name

def convert_to_serializable(obj):
    """
    تحويل الكائنات غير القابلة للتسلسل إلى أنواع قابلة للتسلسل
    """
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    elif isinstance(obj, np.uint8):
        return int(obj)
    else:
        return obj
def analyze_image_colors(image_path):
    """
    تحليل الألوان البارزة في الصورة
    :param image_path: مسار الصورة
    :return: قائمة بالألوان الرئيسية وتوزيعها
    """
    # فتح الصورة
    image = Image.open(image_path)
    
    # تحويل الصورة إلى وضع RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # تقليل حجم الصورة للتحليل السريع
    image = image.resize((150, 150))
    
    # تحويل الصورة إلى مصفوفة
    img_array = np.array(image)
    
    # إعادة تشكيل المصفوفة
    pixels = img_array.reshape(-1, 3)
    
    # حساب الألوان الفريدة وعددها
    color_counts = Counter(map(tuple, pixels))
    
    # استخراج أهم 10 ألوان
    top_colors = color_counts.most_common(10)
    
    # حساب العدد الإجمالي للبكسلات
    total_pixels = sum(count for _, count in top_colors)
    
    # تحليل الألوان
    color_analysis = []
    
    for color, count in top_colors:
        # حساب النسبة المئوية
        percentage = (count / total_pixels) * 100
        
        # تحويل RGB إلى HEX
        hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
        
        # تحديد اسم اللون
        color_name = get_color_name(color)
        
        # تحويل RGB إلى HSV للتحليل
        hsv_color = colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)
        
        color_analysis.append({
            'hex': hex_color,
            'rgb': list(map(int, color)),  # تأكد من تحويل القيم إلى أعداد صحيحة
            'name': color_name,
            'percentage': round(percentage, 2),
            'hue': round(hsv_color[0] * 360, 2),
            'saturation': round(hsv_color[1] * 100, 2),
            'value': round(hsv_color[2] * 100, 2)
        })
    
    return color_analysis

@app.route('/improve', methods=['POST'])
def improve_image():
    print("Received improve request")
    
    # التحقق من وجود الصورة
    if 'image' not in request.files and 'filename' not in request.form:
        print("No image or filename provided")
        return jsonify({'success': False, 'message': 'لم يتم تقديم صورة'})

    try:
        # معالجة رفع الصورة الجديدة
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'success': False, 'message': 'لم يتم اختيار صورة'})
            
            # توليد اسم ملف فريد
            filename = f"upload_{uuid.uuid4().hex[:8]}.png"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        else:
            # استخدام الصورة الموجودة مسبقًا
            filename = request.form.get('filename')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            if not os.path.exists(filepath):
                print("File not found:", filepath)
                return jsonify({'success': False, 'message': 'الملف غير موجود'})

        # استلام قيم أشرطة التمرير
        brightness_value = float(request.form.get('brightness', 20)) / 100 + 1
        contrast_value = float(request.form.get('contrast', 50)) / 100 + 1
        sharpness_value = float(request.form.get('sharpness', 50)) / 100 + 1
        color_value = float(request.form.get('color', 30)) / 100 + 1
        resolution_value = float(request.form.get('resolution', 30)) / 100 + 1
        print("Processing values:", {
            'brightness': brightness_value,
            'contrast': contrast_value,
            'sharpness': sharpness_value,
            'color': color_value,
            'resolution': resolution_value,

        })

        print("Opening image")
        image = Image.open(filepath)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            
        print("Applying enhancements")
        # تحسين الصورة
        enhanced_image = enhance_brightness(image, brightness_value)
        enhanced_image = enhance_contrast(enhanced_image, contrast_value)
        enhanced_image = enhance_sharpness(enhanced_image, sharpness_value)
        enhanced_image = enhance_color(enhanced_image, color_value)
        enhanced_image = enhance_resolution(enhanced_image, resolution_value)

        enhanced_filename = f"enhanced_{filename}"
        enhanced_filepath = os.path.join(app.config['UPLOAD_FOLDER'], enhanced_filename)
        print("Saving enhanced image to:", enhanced_filepath)
        enhanced_image.save(enhanced_filepath)
        
        
        # تحليل الألوان
        color_analysis = analyze_image_colors(filepath)

        # تحويل البيانات للتأكد من إمكانية التسلسل
        serializable_color_analysis = [
            {k: convert_to_serializable(v) for k, v in color.items()} 
            for color in color_analysis
        ]

        return jsonify({
            'success': True,
            'original': filename,
            'enhanced': enhanced_filename,
            'color_analysis': serializable_color_analysis,
            'redirect': url_for('show_result', original=filename, enhanced=enhanced_filename)
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'success': False, 'message': str(e)})
# إضافة route جديد لعرض نتائج الألوان
@app.route('/color_result', methods=['GET', 'POST'])
def color_result():
    if request.method == 'POST':
        # استقبال بيانات تحليل الألوان من الصفحة السابقة
        original_image = request.form.get('original_image')
        color_analysis_str = request.form.get('color_analysis')
        
        # تحويل السلسلة إلى قائمة
        try:
            color_analysis = eval(color_analysis_str)
        except Exception as e:
            print(f"Error parsing color analysis: {e}")
            color_analysis = []
        
        return render_template('result3.html', 
                               original_image=original_image, 
                               color_analysis=color_analysis)
    
    # للطلبات GET
    original_image = request.args.get('original_image')
    color_analysis_str = request.args.get('color_analysis')
    
    try:
        color_analysis = eval(color_analysis_str) if color_analysis_str else []
    except Exception as e:
        print(f"Error parsing color analysis: {e}")
        color_analysis = []
    
    return render_template('result3.html', 
                       original_image=original_image, 
                       color_analysis=color_analysis)
# لعرض نتائج التحسين على صفحة result
@app.route('/result')
def show_result():
    original_image = request.args.get('original')
    enhanced_image = request.args.get('enhanced')
    
    # التحقق من وجود القيم
    if not original_image or not enhanced_image:
        print("Error: Missing original or enhanced image parameters")
        return "خطأ: معلومات الصورة غير مكتملة", 400
    
    # التحقق من أن القيم هي نصوص
    if not isinstance(original_image, str) or not isinstance(enhanced_image, str):
        print("Error: Image parameters must be strings")
        return "خطأ: يجب أن تكون معلومات الصورة نصية", 400
    
    # التحقق من وجود الصور
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_image)
    enhanced_path = os.path.join(app.config['UPLOAD_FOLDER'], enhanced_image)
    
    # طباعة المسارات للتأكد
    print(f"Original path: {original_path}")
    print(f"Enhanced path: {enhanced_path}")
    
    if not os.path.exists(original_path):
        print(f"Original image not found: {original_path}")
        return f"خطأ: الصورة الأصلية {original_image} غير موجودة", 404
    
    if not os.path.exists(enhanced_path):
        print(f"Enhanced image not found: {enhanced_path}")
        return f"خطأ: الصورة المحسنة {enhanced_image} غير موجودة", 404
    
    return render_template('result.html', 
                           original_image=original_image, 
                           enhanced_image=enhanced_image)

# دوال تحسين الصورة المحدثة مع معاملات مخصصة
def enhance_brightness(image, factor):
    """
    تحسين سطوع الصورة بمعامل مخصص
    :param image: الصورة الأصلية
    :param factor: معامل السطوع (1.0 = بدون تغيير)
    """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def enhance_contrast(image, factor):
    """
    تحسين تباين الصورة بمعامل مخصص
    :param image: الصورة الأصلية
    :param factor: معامل التباين (1.0 = بدون تغيير)
    """
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def enhance_sharpness(image, factor):
    """
    تحسين حدة الصورة بمعامل مخصص
    :param image: الصورة الأصلية
    :param factor: معامل الحدة (1.0 = بدون تغيير)
    """
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)

def enhance_color(image, factor):
    """
    تحسين تشبع الألوان بمعامل مخصص
    :param image: الصورة الأصلية
    :param factor: معامل تشبع الألوان (1.0 = بدون تغيير)
    """
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)


def enhance_resolution(image, scale_factor):
    """
    تحسين دقة الصورة
    :param image: الصورة الأصلية
    :param scale_factor: معامل تغيير الحجم
    """
    width, height = image.size
    new_size = (int(width * scale_factor), int(height * scale_factor))
    return image.resize(new_size, Image.LANCZOS)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)