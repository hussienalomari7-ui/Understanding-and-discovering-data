# ============================================================
# Data Cleaning Lab - Part 2 (Educational Version)
# المحاور المغطاة في هذا الملف:
# 1) Missing Values
# 2) Feature Scaling
# 3) Outliers
# ============================================================

# ------------------------------------------------------------
# استيراد المكتبات الأساسية
# ------------------------------------------------------------

# pandas للتعامل مع الجداول والبيانات
import pandas as pd

# numpy للعمليات الرياضية مثل المصفوفات والحسابات
import numpy as np

# seaborn للرسم البياني
import seaborn as sns

# matplotlib لإظهار الرسومات والتحكم فيها
import matplotlib.pyplot as plt

# أدوات التحجيم من sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# مكتبة إحصائية لحساب Z-score
from scipy import stats


# ------------------------------------------------------------
# قراءة البيانات
# ------------------------------------------------------------

# رابط ملف البيانات المستخدم في اللاب
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/Ames_Housing_Data1.tsv"

# قراءة الملف باستخدام pandas
# استخدمنا sep='\t' لأن الملف من نوع TSV وليس CSV
housing = pd.read_csv(url, sep='\t')


# ============================================================
# 1) MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("1) MISSING VALUES")
print("=" * 70)

# ------------------------------------------------------------
# اكتشاف عدد القيم الناقصة في كل عمود
# ------------------------------------------------------------

# isnull() تفحص كل خلية:
# إذا كانت ناقصة ترجع True
# وإذا كانت موجودة ترجع False
#
# sum() تجمع عدد True في كل عمود
# sort_values() ترتب الأعمدة من الأكثر نقصًا إلى الأقل
total_missing = housing.isnull().sum().sort_values(ascending=False)

print("\nعدد القيم الناقصة في أول 20 عمود:")
print(total_missing.head(20))


# ------------------------------------------------------------
# حساب نسبة القيم الناقصة في كل عمود
# ------------------------------------------------------------

# len(housing) يعطي عدد الصفوف الكلي
# نقسم عدد القيم الناقصة على عدد الصفوف ثم نضرب بـ 100
# حتى نحصل على النسبة المئوية
percent_missing = (housing.isnull().sum() / len(housing)) * 100

print("\nنسبة القيم الناقصة في أول 20 عمود:")
print(percent_missing.sort_values(ascending=False).head(20))


# ------------------------------------------------------------
# دمج العدد والنسبة في جدول واحد
# ------------------------------------------------------------

# ننشئ DataFrame جديد يحتوي:
# - عدد القيم الناقصة
# - نسبة القيم الناقصة
missing_data = pd.DataFrame({
    "Total Missing": housing.isnull().sum(),
    "Percent Missing": (housing.isnull().sum() / len(housing)) * 100
})

# نرتب الجدول من الأكثر نقصًا إلى الأقل
missing_data = missing_data.sort_values(by="Total Missing", ascending=False)

print("\nجدول القيم الناقصة (أول 20 عمود):")
print(missing_data.head(20))


# ------------------------------------------------------------
# رسم الأعمدة الأكثر احتواءً على Missing Values
# ------------------------------------------------------------

# نأخذ أول 20 عمود فقط للرسم حتى يكون أوضح
top_missing = total_missing.head(20)

# رسم Bar Chart يوضح الأعمدة التي فيها أكثر قيم ناقصة
top_missing.plot(kind="bar", figsize=(12, 6), fontsize=10)

# تسمية المحاور والعنوان
plt.xlabel("Columns", fontsize=14)
plt.ylabel("Count of Missing Values", fontsize=14)
plt.title("Top 20 Columns with Missing Values", fontsize=16)

# عرض الرسم
plt.show()


# ------------------------------------------------------------
# الطريقة 1: حذف الصفوف التي تحتوي على Missing Values
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("الطريقة 1: حذف الصفوف التي تحتوي على Missing Values")
print("-" * 70)

# dropna(subset=[...]) يعني:
# احذف الصف إذا كان العمود المحدد يحتوي على قيمة ناقصة
drop_rows_example = housing.dropna(subset=["Lot Frontage"])

print("شكل الجدول الأصلي:")
print(housing.shape)

print("شكل الجدول بعد حذف الصفوف الناقصة في Lot Frontage:")
print(drop_rows_example.shape)


# ------------------------------------------------------------
# الطريقة 2: حذف العمود كاملًا
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("الطريقة 2: حذف عمود كامل")
print("-" * 70)

# drop("Lot Frontage", axis=1) يعني:
# احذف العمود بالكامل
# axis=1 تعني أننا نحذف عمودًا وليس صفًا
drop_column_example = housing.drop("Lot Frontage", axis=1)

print("شكل الجدول الأصلي:")
print(housing.shape)

print("شكل الجدول بعد حذف عمود Lot Frontage:")
print(drop_column_example.shape)


# ------------------------------------------------------------
# الطريقة 3: تعويض القيم الناقصة بالوسيط Median
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("الطريقة 3: تعويض Missing Values باستخدام Median")
print("-" * 70)

# نحسب الوسيط لعمود Lot Frontage
lot_frontage_median = housing["Lot Frontage"].median()

print("وسيط Lot Frontage:")
print(lot_frontage_median)

# نملأ القيم الناقصة في العمود باستخدام الوسيط
# inplace=True يعني عدّل على الجدول نفسه مباشرة
housing["Lot Frontage"].fillna(lot_frontage_median, inplace=True)

# نتحقق هل بقيت قيم ناقصة في هذا العمود أم لا
print("عدد القيم الناقصة المتبقية في Lot Frontage:")
print(housing["Lot Frontage"].isnull().sum())


# ------------------------------------------------------------
# Exercise: تعويض Mas Vnr Area بالمتوسط Mean
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("Exercise: تعويض Mas Vnr Area باستخدام Mean")
print("-" * 70)

# نحسب المتوسط الحسابي للعمود
mas_vnr_mean = housing["Mas Vnr Area"].mean()

print("متوسط Mas Vnr Area:")
print(mas_vnr_mean)

# نملأ القيم الناقصة بالمتوسط
housing["Mas Vnr Area"].fillna(mas_vnr_mean, inplace=True)

# نتحقق من أن القيم الناقصة اختفت
print("عدد القيم الناقصة المتبقية في Mas Vnr Area:")
print(housing["Mas Vnr Area"].isnull().sum())


# ============================================================
# 2) FEATURE SCALING
# ============================================================

print("\n" + "=" * 70)
print("2) FEATURE SCALING")
print("=" * 70)

# ------------------------------------------------------------
# اختيار الأعمدة الرقمية فقط
# ------------------------------------------------------------

# select_dtypes(include=[...]) يختار فقط الأعمدة ذات النوع الرقمي
# لأن التحجيم يطبق على الأرقام فقط
hous_num = housing.select_dtypes(include=["int64", "float64"])

print("\nشكل الأعمدة الرقمية فقط:")
print(hous_num.shape)

print("\nأول 5 صفوف من الأعمدة الرقمية:")
print(hous_num.head())


# ------------------------------------------------------------
# MinMaxScaler
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("MinMaxScaler")
print("-" * 70)

# MinMaxScaler يحول كل عمود رقمي بحيث تصبح قيمه بين 0 و 1 تقريبًا
# أصغر قيمة في العمود تصبح 0
# أكبر قيمة في العمود تصبح 1
minmax_scaler = MinMaxScaler()

# fit_transform():
# fit = يتعلم أقل وأكبر قيمة من البيانات
# transform = يطبق التحويل
norm_data = minmax_scaler.fit_transform(hous_num)

print("أول 5 صفوف بعد MinMax Scaling:")
print(norm_data[:5])


# ------------------------------------------------------------
# StandardScaler
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("StandardScaler")
print("-" * 70)

# StandardScaler يحول البيانات بحيث:
# المتوسط يصبح 0
# والانحراف المعياري يصبح 1
standard_scaler = StandardScaler()

# نطبق التحويل على الأعمدة الرقمية
scaled_data = standard_scaler.fit_transform(hous_num)

print("أول 5 صفوف بعد Standard Scaling:")
print(scaled_data[:5])


# ------------------------------------------------------------
# Exercise: Standardization لعمود SalePrice فقط
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("Exercise: Standardize SalePrice only")
print("-" * 70)

# housing[["SalePrice"]] يرجع DataFrame بعمود واحد
# وهذا الشكل مناسب لـ StandardScaler
scaled_sprice = StandardScaler().fit_transform(housing[["SalePrice"]])

print("أول 5 قيم من SalePrice بعد Standardization:")
print(scaled_sprice[:5])


# ============================================================
# 3) OUTLIERS
# ============================================================

print("\n" + "=" * 70)
print("3) OUTLIERS")
print("=" * 70)

# ------------------------------------------------------------
# Univariate Analysis باستخدام Boxplot
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("Boxplot for Lot Area")
print("-" * 70)

# Boxplot يساعدنا في اكتشاف القيم الشاذة بصريًا
# النقاط الموجودة بعيدًا عن الصندوق غالبًا تكون Outliers
plt.figure(figsize=(10, 5))
sns.boxplot(x=housing["Lot Area"])
plt.title("Boxplot of Lot Area")
plt.xlabel("Lot Area")
plt.show()


print("\n" + "-" * 70)
print("Boxplot for SalePrice")
print("-" * 70)

# رسم Boxplot لعمود SalePrice
plt.figure(figsize=(10, 5))
sns.boxplot(x=housing["SalePrice"])
plt.title("Boxplot of SalePrice")
plt.xlabel("SalePrice")
plt.show()


# ------------------------------------------------------------
# Bivariate Analysis باستخدام Scatter Plot
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("Scatter Plot: Gr Liv Area vs SalePrice")
print("-" * 70)

# scatter plot يرسم العلاقة بين عمودين
# وكل نقطة تمثل سجلًا واحدًا (بيتًا واحدًا)
plt.figure(figsize=(8, 5))
housing.plot.scatter(x="Gr Liv Area", y="SalePrice")
plt.title("Gr Liv Area vs SalePrice")
plt.show()


# ------------------------------------------------------------
# عرض أكبر القيم في Gr Liv Area
# ------------------------------------------------------------

# نرتب البيانات تنازليًا حسب Gr Liv Area
# حتى نرى الصفوف الأكبر والأكثر احتمالًا أن تكون Outliers
print("\nأكبر 5 قيم في Gr Liv Area:")
print(housing.sort_values(by="Gr Liv Area", ascending=False).head(5))


# ------------------------------------------------------------
# حذف Outliers محددة في Gr Liv Area
# ------------------------------------------------------------

# في اللاب تم حذف صفين محددين اعتبرا قيمًا شاذة
# نحتفظ بالناتج في DataFrame جديد بدل تعديل الجدول الأصلي مباشرة
outliers_dropped = housing.drop(housing.index[[1499, 2181]])

print("\nشكل الجدول قبل حذف الـ outliers:")
print(housing.shape)

print("شكل الجدول بعد حذف الـ outliers من Gr Liv Area:")
print(outliers_dropped.shape)


# ------------------------------------------------------------
# Outliers إعادة الرسم بعد حذف الـ 
# ------------------------------------------------------------

# نعيد رسم العلاقة لنرى الفرق بعد إزالة القيم الشاذة
plt.figure(figsize=(8, 5))
outliers_dropped.plot.scatter(x="Gr Liv Area", y="SalePrice")
plt.title("Gr Liv Area vs SalePrice (After Removing Outliers)")
plt.show()


# ------------------------------------------------------------
# Z-score على عمود Low Qual Fin SF
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("Z-score for Low Qual Fin SF")
print("-" * 70)

# zscore() تحسب بعد كل قيمة عن المتوسط بوحدة الانحراف المعياري
# إذا كانت القيمة أكبر من 3 أو أقل من -3 غالبًا تعتبر Outlier
housing["LQFSF_Stats"] = stats.zscore(housing["Low Qual Fin SF"])

# نعرض ملخصًا للعمود الأصلي وعمود Z-score
print("ملخص Low Qual Fin SF مع Z-score:")
print(housing[["Low Qual Fin SF", "LQFSF_Stats"]].describe().round(3))


# ------------------------------------------------------------
# استخراج القيم التي Z-score لها أكبر من 3
# ------------------------------------------------------------

# abs(...) > 3 يعني:
# خذ القيم التي تبعد أكثر من 3 انحرافات معيارية عن المتوسط
outliers_z = housing[abs(housing["LQFSF_Stats"]) > 3]

print("\nعدد الـ outliers في Low Qual Fin SF:")
print(outliers_z.shape)

print("\nأول الصفوف التي تعتبر outliers في Low Qual Fin SF:")
print(outliers_z[["Low Qual Fin SF", "LQFSF_Stats"]].head())


# ------------------------------------------------------------
# Exercise: تحليل Outliers في Lot Area
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("Exercise: Analyze Outliers in Lot Area")
print("-" * 70)

# Boxplot لعمود Lot Area
plt.figure(figsize=(10, 5))
sns.boxplot(x=housing["Lot Area"])
plt.title("Boxplot of Lot Area")
plt.xlabel("Lot Area")
plt.show()

# Scatter plot بين Lot Area و SalePrice
plt.figure(figsize=(8, 5))
housing.plot.scatter(x="Lot Area", y="SalePrice")
plt.title("Lot Area vs SalePrice")
plt.show()

# حساب Z-score لعمود Lot Area
housing["Lot_Area_Stats"] = stats.zscore(housing["Lot Area"])

# عرض ملخص العمود مع Z-score
print("ملخص Lot Area مع Z-score:")
print(housing[["Lot Area", "Lot_Area_Stats"]].describe().round(3))

# عرض أكبر قيمة في Lot Area
print("\nأكبر قيمة في Lot Area:")
print(housing.sort_values(by="Lot Area", ascending=False).head(1))


# ------------------------------------------------------------
# مثال على حذف صف معين اعتبر Outlier في Lot Area
# ------------------------------------------------------------

# هذا مثال من اللاب على حذف صف معين
lot_area_removed = housing.drop(housing.index[[957]])

print("\nشكل الجدول قبل حذف Outlier من Lot Area:")
print(housing.shape)

print("شكل الجدول بعد حذف Outlier من Lot Area:")
print(lot_area_removed.shape)


