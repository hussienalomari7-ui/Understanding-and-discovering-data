# =========================================================
# PART 1: Data Loading, Initial Cleaning, Encoding, Skew Fix
# =========================================================

import numpy as np                      # استيراد numpy للحسابات الرياضية والتحويلات العددية
import pandas as pd                     # استيراد pandas للتعامل مع الجداول والبيانات
import seaborn as sns                   # استيراد seaborn للرسم الإحصائي
import matplotlib.pyplot as plt         # استيراد matplotlib للرسم البياني الأساسي

sns.set_theme()                         # تحسين الشكل العام للرسومات

# ---------------------------------------------------------
# 1) Load the dataset
# ---------------------------------------------------------

datafile = "data/Ames_Housing_Data.tsv" # مسار ملف البيانات؛ غيّره حسب مكان الملف عندك
df = pd.read_csv(datafile, sep="\t")    # قراءة الملف بصيغة TSV لأن الفاصل هو Tab

print("Initial shape:", df.shape)       # طباعة عدد الصفوف والأعمدة قبل أي تعديل
print(df.head())                        # عرض أول 5 صفوف للتأكد أن القراءة تمت بشكل صحيح

# ---------------------------------------------------------
# 2) Inspect the dataset structure
# ---------------------------------------------------------

print(df.info())                        # عرض أنواع الأعمدة وعدد القيم غير المفقودة بكل عمود
print(df.describe().T.head())           # عرض ملخص إحصائي سريع لأول مجموعة من الأعمدة الرقمية

# ---------------------------------------------------------
# 3) Remove obvious outliers used in the lab
# ---------------------------------------------------------

df = df.loc[df["Gr Liv Area"] <= 4000, :]   # الاحتفاظ فقط بالمنازل التي مساحة المعيشة فيها <= 4000
print("Shape after outlier removal:", df.shape)  # معرفة شكل البيانات بعد حذف القيم الشاذة

# ---------------------------------------------------------
# 4) Keep a clean copy for later experiments
# ---------------------------------------------------------

data = df.copy()                        # إنشاء نسخة مرجعية من البيانات بعد التنظيف الأولي

# ---------------------------------------------------------
# 5) Identify categorical columns
# ---------------------------------------------------------

categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()  # استخراج أسماء الأعمدة الفئوية
print("Categorical columns count:", len(categorical_cols))                           # طباعة عدد الأعمدة الفئوية
print("Sample categorical columns:", categorical_cols[:10])                         # عرض أول 10 أعمدة فئوية

# ---------------------------------------------------------
# 6) Preview categorical columns before encoding
# ---------------------------------------------------------

print(df[categorical_cols].head().T)    # عرض أول القيم للأعمدة الفئوية بشكل عمودي لتسهيل قراءتها

# ---------------------------------------------------------
# 7) Convert categorical variables to dummy variables
# ---------------------------------------------------------

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)  # تحويل الأعمدة الفئوية إلى dummy variables
print("Encoded shape:", df_encoded.shape)                                    # عرض شكل البيانات بعد التوسيع بالأعمدة الجديدة
print(df_encoded.head())                                                     # عرض أول صفوف بعد التحويل

# ---------------------------------------------------------
# 8) Detect skewed numeric columns
# ---------------------------------------------------------

float_cols = data.select_dtypes(include=["float64", "float32"]).columns      # تحديد الأعمدة الرقمية المستمرة من نوع float
skew_vals = data[float_cols].skew().sort_values(ascending=False)             # حساب skewness لكل عمود وترتيبها تنازليًا
skew_cols = skew_vals[skew_vals > 0.75]                                      # الاحتفاظ فقط بالأعمدة التي skewness فيها أكبر من 0.75

print("Highly skewed columns:")                                              # عنوان توضيحي
print(skew_cols)                                                             # عرض الأعمدة الأكثر انحرافًا

# ---------------------------------------------------------
# 9) Visualize effect of log1p on one skewed feature
# ---------------------------------------------------------

feature = "Misc Val"                                                         # اختيار عمود من الأعمدة skewed لعرض المثال عليه

fig, ax = plt.subplots(1, 2, figsize=(12, 4))                                # إنشاء شكل فيه رسمتان جنبًا إلى جنب

sns.histplot(data[feature], kde=True, ax=ax[0])                              # رسم توزيع العمود الأصلي
ax[0].set_title(f"Original: {feature}")                                      # وضع عنوان للرسم الأول

sns.histplot(np.log1p(data[feature]), kde=True, ax=ax[1])                    # رسم التوزيع بعد log1p
ax[1].set_title(f"Log1p transformed: {feature}")                             # وضع عنوان للرسم الثاني

plt.tight_layout()                                                           # ترتيب الشكل حتى لا تتداخل العناصر
plt.show()                                                                   # عرض الرسم

# ---------------------------------------------------------
# 10) Apply log1p to all highly skewed columns
# ---------------------------------------------------------

df_model = df.copy()                                                         # إنشاء نسخة عمل جديدة سنطبق عليها التحويلات

for col in skew_cols.index:                                                  # المرور على كل عمود skewed
    df_model[col] = np.log1p(df_model[col])                                  # تطبيق log1p لتخفيف الانحراف في التوزيع

print("Shape after skew correction:", df_model.shape)                        # التأكد أن عدد الصفوف والأعمدة لم يتغير
print(df_model.head())                                                       # عرض أول الصفوف بعد التحويل

# ---------------------------------------------------------
# 11) Build a smaller educational subset for feature engineering
# ---------------------------------------------------------

smaller_df = data.loc[:, [                                                   # اختيار مجموعة صغيرة وواضحة من الأعمدة للتجارب التعليمية
    "Lot Area",                                                              # مساحة الأرض
    "Overall Qual",                                                          # الجودة العامة
    "Overall Cond",                                                          # الحالة العامة
    "Year Built",                                                            # سنة البناء
    "Year Remod/Add",                                                        # سنة التعديل أو التجديد
    "Gr Liv Area",                                                           # مساحة المعيشة
    "Full Bath",                                                             # عدد الحمامات الكاملة
    "Bedroom AbvGr",                                                         # عدد غرف النوم فوق الأرض
    "TotRms AbvGrd",                                                         # عدد الغرف فوق الأرض
    "Garage Cars",                                                           # سعة الكراج بعدد السيارات
    "Garage Area",                                                           # مساحة الكراج
    "SalePrice"                                                              # سعر البيع وهو الهدف
]].copy()                                                                    # أخذ نسخة مستقلة من هذا subset

smaller_df = smaller_df.fillna(0)                                            # تعويض القيم المفقودة بصفر لتسهيل التجارب التعليمية

print(smaller_df.info())                                                     # التأكد من أنواع الأعمدة وعدم وجود missing values
print(smaller_df.describe().T)                                               # عرض ملخص إحصائي كامل للـ subset

# ---------------------------------------------------------
# 12) Optional pairplot for intuition before feature engineering
# ---------------------------------------------------------

sns.pairplot(smaller_df, plot_kws={"alpha": 0.1, "edgecolor": "none"})       # رسم العلاقات الثنائية بين الميزات والهدف
plt.show()                                                                   # عرض الشكل

