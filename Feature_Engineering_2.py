# =========================================================
# PART 2: Manual Feature Engineering + Category-Based Features
# =========================================================

import numpy as np                                      # استيراد numpy لاستخدام العمليات الرياضية
import pandas as pd                                     # استيراد pandas للتعامل مع الجداول
from sklearn.preprocessing import PolynomialFeatures    # استيراد أداة توليد polynomial features من sklearn

# ---------------------------------------------------------
# 1) Manual polynomial features
# ---------------------------------------------------------

X2 = X.copy()                                           # إنشاء نسخة من X للحفاظ على الأصل بدون تعديل

X2["OQ2"] = X2["Overall Qual"] ** 2                     # إنشاء مربع الجودة العامة لالتقاط علاقة غير خطية
X2["GLA2"] = X2["Gr Liv Area"] ** 2                     # إنشاء مربع مساحة المعيشة لالتقاط الانحناء في العلاقة

print(X2.head())                                        # عرض أول الصفوف للتأكد من الأعمدة الجديدة

# ---------------------------------------------------------
# 2) Manual interaction features
# ---------------------------------------------------------

X3 = X2.copy()                                          # إنشاء نسخة جديدة فوق النسخة التي فيها التربيعات

X3["OQ_x_YB"] = X3["Overall Qual"] * X3["Year Built"]   # تفاعل بين الجودة العامة وسنة البناء
X3["OQ_/_LA"] = X3["Overall Qual"] / X3["Lot Area"]     # نسبة الجودة العامة إلى مساحة الأرض

print(X3.head())                                        # عرض أول الصفوف بعد إضافة interaction features
print(X3.info())                                        # التحقق من شكل البيانات وأنواع الأعمدة

# ---------------------------------------------------------
# 3) Explore House Style as a categorical feature
# ---------------------------------------------------------

print(data["House Style"].value_counts())               # عدّ تكرار كل نوع من أنواع المنازل

house_style_dummies = pd.get_dummies(                   # إنشاء dummy variables لعمود House Style كمثال توضيحي
    data["House Style"],                                # العمود الفئوي المطلوب تحويله
    drop_first=True                                     # حذف أول فئة لتجنب التكرار الخطي
)

print(house_style_dummies.head())                       # عرض أول صفوف الـ dummies الناتجة

# ---------------------------------------------------------
# 4) Prepare Neighborhood categories and merge rare ones
# ---------------------------------------------------------

nbh_counts = data["Neighborhood"].value_counts()        # عدّ تكرار كل حي داخل البيانات
print(nbh_counts)                                       # عرض عدد البيوت بكل حي

other_nbhs = list(nbh_counts[nbh_counts <= 8].index)    # جمع أسماء الأحياء النادرة التي ظهورها <= 8
print("Rare neighborhoods merged into 'Other':")        # عنوان توضيحي
print(other_nbhs)                                       # عرض قائمة الأحياء النادرة

# ---------------------------------------------------------
# 5) Add categorical columns to feature table before encoding
# ---------------------------------------------------------

X4 = X3.copy()                                          # إنشاء نسخة جديدة تتضمن الميزات اليدوية السابقة

X4["Neighborhood"] = data["Neighborhood"].replace(      # إنشاء عمود حي جديد بعد دمج الأحياء النادرة
    other_nbhs,                                         # قائمة الأحياء النادرة التي سيتم استبدالها
    "Other"                                             # الاسم البديل الموحد للفئات النادرة
)

X4["House Style"] = data["House Style"]                 # إضافة نوع المنزل كما هو قبل تحويله إلى dummies

print(X4[["Neighborhood", "House Style"]].head())       # عرض أول القيم للفئتين للتأكد من إضافتهما

# ---------------------------------------------------------
# 6) One-hot encode the categorical features
# ---------------------------------------------------------

X4_encoded = pd.get_dummies(                            # تحويل الأعمدة الفئوية إلى dummy variables
    X4,                                                 # جدول الميزات الحالي
    columns=["Neighborhood", "House Style"],            # الأعمدة الفئوية التي سنحوّلها
    drop_first=True                                     # حذف أول فئة من كل عمود فئوي
)

print("X4_encoded shape:", X4_encoded.shape)            # عرض أبعاد الجدول بعد التوسيع
print(X4_encoded.head())                                # عرض أول الصفوف بعد التحويل

# ---------------------------------------------------------
# 7) Category aggregate deviation features
# ---------------------------------------------------------

X5 = X3.copy()                                          # إنشاء نسخة جديدة من الميزات العددية الهندسية
X5["Neighborhood"] = data["Neighborhood"].replace(other_nbhs, "Other")  # إضافة Neighborhood بعد دمج النادر
X5["House Style"] = data["House Style"]                 # إضافة House Style لاستخدامها كمجموعة مرجعية

def add_deviation_feature(X, feature, category):        # تعريف دالة تحسب انحراف القيمة عن متوسط مجموعتها
    category_mean = X.groupby(category)[feature].transform("mean")  # حساب متوسط feature داخل كل فئة وإرجاعه لكل صف
    return X[feature] - category_mean                   # طرح متوسط المجموعة من القيمة الأصلية

X5["dev_LotArea_by_Neighborhood"] = add_deviation_feature(   # ميزة: كم تختلف مساحة الأرض عن متوسط الحي
    X5,                                                        # جدول البيانات الحالي
    "Lot Area",                                                # العمود العددي المستهدف
    "Neighborhood"                                             # العمود الفئوي المرجعي
)

X5["dev_GrLivArea_by_Neighborhood"] = add_deviation_feature(  # ميزة: كم تختلف مساحة المعيشة عن متوسط الحي
    X5,                                                        # جدول البيانات الحالي
    "Gr Liv Area",                                             # العمود العددي المستهدف
    "Neighborhood"                                             # العمود الفئوي المرجعي
)

X5["dev_TotRms_by_HouseStyle"] = add_deviation_feature(       # ميزة: كم يختلف عدد الغرف عن متوسط نوع المنزل
    X5,                                                        # جدول البيانات الحالي
    "TotRms AbvGrd",                                           # العمود العددي المستهدف
    "House Style"                                              # العمود الفئوي المرجعي
)

print(X5[[
    "Neighborhood",                                            # الحي
    "House Style",                                             # نوع المنزل
    "Lot Area",                                                # مساحة الأرض الأصلية
    "Gr Liv Area",                                             # مساحة المعيشة الأصلية
    "TotRms AbvGrd",                                           # عدد الغرف الأصلية
    "dev_LotArea_by_Neighborhood",                             # انحراف مساحة الأرض عن متوسط الحي
    "dev_GrLivArea_by_Neighborhood",                           # انحراف مساحة المعيشة عن متوسط الحي
    "dev_TotRms_by_HouseStyle"                                 # انحراف عدد الغرف عن متوسط نوع المنزل
]].head())                                                     # عرض أول صفوف لفهم الميزات الجديدة

# ---------------------------------------------------------
# 8) Encode categorical columns if X5 will be used in a model
# ---------------------------------------------------------

X5_model = pd.get_dummies(                               # تحويل الأعمدة الفئوية داخل X5 إلى أعمدة رقمية
    X5,                                                  # جدول البيانات الحالي
    columns=["Neighborhood", "House Style"],             # الأعمدة النصية التي يجب تحويلها
    drop_first=True                                      # حذف أول فئة من كل عمود
)

print("X5_model shape:", X5_model.shape)                 # عرض شكل البيانات الجاهزة للنمذجة
print(X5_model.head())                                   # عرض أول الصفوف بعد التحويل

# ---------------------------------------------------------
# 9) Automatic polynomial features with sklearn
# ---------------------------------------------------------

poly_features = X[["Overall Qual", "Gr Liv Area", "Garage Area"]].copy()  # اختيار 3 أعمدة فقط لتجربة PolynomialFeatures
print(poly_features.head())                                                # عرض الأعمدة المختارة قبل التوليد

poly = PolynomialFeatures(                         # إنشاء مولد features polynomial
    degree=2,                                     # حتى الدرجة الثانية: أعمدة أصلية + مربعات + interactions
    include_bias=False                             # عدم إضافة عمود ثابت كله 1
)

poly_array = poly.fit_transform(poly_features)     # تطبيق التوليد على الأعمدة المختارة

poly_feature_names = poly.get_feature_names_out(   # استخراج أسماء الأعمدة الجديدة الناتجة
    poly_features.columns                          # تمرير أسماء الأعمدة الأصلية
)

poly_df = pd.DataFrame(                            # تحويل المصفوفة الناتجة إلى DataFrame واضح
    poly_array,                                    # البيانات الناتجة من PolynomialFeatures
    columns=poly_feature_names,                    # أسماء الأعمدة الجديدة
    index=poly_features.index                      # الحفاظ على نفس الفهرس الأصلي
)

print("Polynomial feature names:")                 # عنوان توضيحي
print(poly_feature_names)                          # عرض أسماء الميزات المولدة آليًا
print(poly_df.head())                              # عرض أول الصفوف من الجدول الجديد
