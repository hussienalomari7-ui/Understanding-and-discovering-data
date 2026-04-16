# =========================================
# Data Cleaning Lab - Educational Version
# هذا الملف التعليمي يغطي الأفكار التي تعلمناها:
# 1) استيراد المكتبات
# 2) تنزيل/قراءة البيانات
# 3) استعراض أول الصفوف
# 4) معرفة حجم البيانات
# 5) فهم بنية البيانات باستخدام info()
# 6) تلخيص عمود رقمي باستخدام describe()
# 7) تلخيص عمود نصي باستخدام value_counts()
# 8) استخراج الأعمدة الرقمية
# 9) حساب الارتباط Correlation مع SalePrice
# 10) رسم التوزيع وفهم الالتواء Skewness
# 11) تطبيق Log Transform
# 12) اكتشاف التكرار وحذفه
# =========================================

# استيراد مكتبة pandas للتعامل مع الجداول
import pandas as pd

# استيراد مكتبة numpy للعمليات الرياضية مثل log
import numpy as np

# استيراد seaborn للرسم البياني
import seaborn as sns

# استيراد matplotlib لإظهار الرسومات
import matplotlib.pyplot as plt


# =========================================
# 1) قراءة البيانات
# =========================================

# هذا هو رابط ملف البيانات الصحيح من اللاب
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/Ames_Housing_Data1.tsv"

# نقرأ الملف باستخدام pandas
# sep='\t' لأن الملف من نوع TSV وليس CSV
housing = pd.read_csv(url, sep='\t')


# =========================================
# 2) استعراض أول البيانات
# =========================================

print("\n" + "=" * 60)
print("أول 5 صفوف من البيانات:")
print("=" * 60)

# head() تعرض أول 5 صفوف من الجدول
print(housing.head())


# =========================================
# 3) معرفة حجم البيانات
# =========================================

print("\n" + "=" * 60)
print("حجم البيانات:")
print("=" * 60)

# shape تعطي عدد الصفوف وعدد الأعمدة
print(housing.shape)

# يمكن أيضًا تخزين عدد الصفوف والأعمدة بشكل منفصل
rows, cols = housing.shape
print(f"عدد الصفوف = {rows}")
print(f"عدد الأعمدة = {cols}")


# =========================================
# 4) فهم هيكل البيانات
# =========================================

print("\n" + "=" * 60)
print("معلومات عامة عن البيانات باستخدام info():")
print("=" * 60)

# info() تعرض:
# - عدد الصفوف
# - عدد الأعمدة
# - أسماء الأعمدة
# - عدد القيم غير الفارغة في كل عمود
# - نوع البيانات لكل عمود
housing.info()


# =========================================
# 5) تلخيص عمود رقمي
# =========================================

print("\n" + "=" * 60)
print("ملخص إحصائي لعمود SalePrice:")
print("=" * 60)

# describe() تعطي ملخصًا إحصائيًا للعمود الرقمي
# مثل count, mean, std, min, max وغيرها
print(housing["SalePrice"].describe())


# =========================================
# 6) تلخيص عمود نصي/فئوي
# =========================================

print("\n" + "=" * 60)
print("عدد مرات ظهور كل قيمة في Sale Condition:")
print("=" * 60)

# value_counts() تحسب عدد تكرار كل فئة داخل العمود النصي
print(housing["Sale Condition"].value_counts())


# =========================================
# 7) اختيار الأعمدة الرقمية فقط
# =========================================

print("\n" + "=" * 60)
print("الأعمدة الرقمية فقط:")
print("=" * 60)

# select_dtypes() تختار الأعمدة بحسب نوع البيانات
# هنا نريد فقط الأعمدة الرقمية: int64 و float64
hous_num = housing.select_dtypes(include=["int64", "float64"])

# نعرض أول 5 صفوف من الأعمدة الرقمية فقط
print(hous_num.head())


# =========================================
# 8) حساب Correlation
# =========================================

print("\n" + "=" * 60)
print("Correlation مع SalePrice:")
print("=" * 60)

# corr() تنشئ مصفوفة ارتباط بين جميع الأعمدة الرقمية
corr_matrix = hous_num.corr()

# نأخذ فقط الارتباطات الخاصة بعمود SalePrice
saleprice_corr = corr_matrix["SalePrice"]

# نرتب النتائج من الأعلى إلى الأقل
print(saleprice_corr.sort_values(ascending=False))


# =========================================
# 9) عرض الارتباطات القوية فقط
# =========================================

print("\n" + "=" * 60)
print("الأعمدة ذات الارتباط القوي مع SalePrice (أكبر من 0.5):")
print("=" * 60)

# abs() تأخذ القيمة المطلقة حتى نلتقط الارتباط القوي
# سواء كان موجبًا أو سالبًا
strong_corr = saleprice_corr[abs(saleprice_corr) > 0.5]

# عرض النتائج مرتبة
print(strong_corr.sort_values(ascending=False))


# =========================================
# 10) رسم توزيع SalePrice قبل التحويل
# =========================================

# نفتح نافذة رسم جديدة بحجم مناسب
plt.figure(figsize=(8, 5))

# histplot يرسم هيستوغرام
# kde=True يرسم خطًا ناعمًا لتوضيح شكل التوزيع
sns.histplot(housing["SalePrice"], kde=True)

# عنوان للرسم
plt.title("Distribution of SalePrice")

# اسم المحور الأفقي
plt.xlabel("SalePrice")

# إظهار الرسم
plt.show()


# =========================================
# 11) حساب Skewness قبل التحويل
# =========================================

print("\n" + "=" * 60)
print("Skewness قبل Log Transform:")
print("=" * 60)

# skew() تقيس ميل التوزيع
# إذا كانت القيمة موجبة وكبيرة نسبيًا => التوزيع مائل لليمين
original_skew = housing["SalePrice"].skew()
print(original_skew)


# =========================================
# 12) تطبيق Log Transform
# =========================================

# نطبق اللوغاريتم الطبيعي على SalePrice
# هذا يساعد في تقليل الالتواء وجعل التوزيع أقرب للطبيعي
log_saleprice = np.log(housing["SalePrice"])

print("\n" + "=" * 60)
print("أول 5 قيم من log(SalePrice):")
print("=" * 60)

# عرض أول 5 قيم بعد التحويل
print(log_saleprice.head())


# =========================================
# 13) رسم توزيع SalePrice بعد التحويل اللوغاريتمي
# =========================================

plt.figure(figsize=(8, 5))
sns.histplot(log_saleprice, kde=True)
plt.title("Distribution of log(SalePrice)")
plt.xlabel("log(SalePrice)")
plt.show()


# =========================================
# 14) حساب Skewness بعد Log Transform
# =========================================

print("\n" + "=" * 60)
print("Skewness بعد Log Transform:")
print("=" * 60)

# نحسب الالتواء بعد التحويل
log_skew = log_saleprice.skew()
print(log_skew)

# مقارنة قبل وبعد
print("\nمقارنة الالتواء:")
print(f"قبل التحويل  = {original_skew}")
print(f"بعد التحويل  = {log_skew}")


# =========================================
# 15) اكتشاف التكرار باستخدام PID
# =========================================

print("\n" + "=" * 60)
print("الصفوف التي تحتوي PID مكرر:")
print("=" * 60)

# duplicated(["PID"]) تفحص هل قيمة PID تكررت من قبل
# housing[...] ترجع فقط الصفوف المكررة
duplicate_pid = housing[housing.duplicated(["PID"])]

# shape تعطي عدد الصفوف والأعمدة في البيانات المكررة
print(duplicate_pid.shape)

# عرض أول الصفوف المكررة
print(duplicate_pid.head())

# =========================================
# 16) حذف الصفوف المكررة بالكامل
# =========================================

print("\n" + "=" * 60)
print("حذف الصفوف المكررة بالكامل:")
print("=" * 60)

# drop_duplicates() يحذف الصفوف المتطابقة المكررة بالكامل
housing_no_duplicates = housing.drop_duplicates()

# مقارنة حجم الجدول قبل وبعد الحذف
print("حجم الجدول الأصلي:")
print(housing.shape)

print("حجم الجدول بعد حذف الصفوف المكررة بالكامل:")
print(housing_no_duplicates.shape)

# =========================================
# 17) حذف التكرار بناءً على عمود محدد
# =========================================

print("\n" + "=" * 60)
print("حذف التكرار بناءً على عمود Order فقط:")
print("=" * 60)

# subset=["Order"] يعني:
# افحص التكرار فقط في عمود Order
housing_order_unique = housing.drop_duplicates(subset=["Order"])

# مقارنة حجم الجدول قبل وبعد
print("حجم الجدول الأصلي:")
print(housing.shape)

print("حجم الجدول بعد حذف التكرار اعتمادًا على Order:")
print(housing_order_unique.shape)


