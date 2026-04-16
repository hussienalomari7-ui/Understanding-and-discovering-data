# =========================================================
# الجزء الثاني: التصفية + groupby + visualization
# =========================================================

# -----------------------------
# 1) استيراد المكتبات المطلوبة
# -----------------------------
import pandas as pd                  # لمعالجة البيانات
import plotly.express as px          # للرسم البياني
import requests                      # لجلب ملف geojson من الإنترنت
import json                          # لقراءة ملف geojson النصي

# -----------------------------------------
# 2) قراءة البيانات وتجهيزها من جديد
#    (حتى يكون هذا الجزء مستقلًا بذاته)
# -----------------------------------------
data_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/18100001.csv"
gasoline = pd.read_csv(data_url)     # قراءة ملف البيانات الأصلي

data = gasoline[['REF_DATE', 'GEO', 'Type of fuel', 'VALUE']].rename(
    columns={
        'REF_DATE': 'DATE',          # إعادة تسمية العمود الزمني
        'Type of fuel': 'TYPE'       # إعادة تسمية نوع الوقود
    }
)

data[['City', 'Province']] = data['GEO'].str.split(',', n=1, expand=True)  # فصل المدينة عن المقاطعة
data['City'] = data['City'].str.strip()          # إزالة المسافات الزائدة
data['Province'] = data['Province'].str.strip()  # إزالة المسافات الزائدة
data['DATE'] = pd.to_datetime(data['DATE'], format='%b-%y')  # تحويل التاريخ
data['Month'] = data['DATE'].dt.month_name().str[:3]         # استخراج الشهر
data['Year'] = data['DATE'].dt.year                          # استخراج السنة

# =========================================================
# 3) أمثلة على التصفية البسيطة
# =========================================================

# مثال 1: اختيار بيانات Calgary فقط
calgary = data[data['GEO'] == 'Calgary, Alberta']   # فلترة جميع الصفوف الخاصة بمدينة Calgary
print("بيانات Calgary:")
print(calgary.head())

# مثال 2: اختيار بيانات سنة 2000 فقط
year_2000 = data[data['Year'] == 2000]              # فلترة الصفوف التي سنة تاريخها 2000
print("\nبيانات سنة 2000:")
print(year_2000.head())

# =========================================================
# 4) التصفية باستخدام أكثر من شرط
# =========================================================

# مثال 3: اختيار Toronto و Edmonton
two_cities = data[
    (data['GEO'] == 'Toronto, Ontario') |           # الشرط الأول: Toronto
    (data['GEO'] == 'Edmonton, Alberta')            # الشرط الثاني: Edmonton
]
print("\nبيانات Toronto و Edmonton:")
print(two_cities.head())

# مثال 4: استخدام isin بدل تكرار OR
cities = ['Calgary', 'Toronto', 'Edmonton']         # قائمة المدن المطلوبة
selected_cities = data[data['City'].isin(cities)]   # اختيار الصفوف التي المدينة فيها ضمن القائمة
print("\nبيانات Calgary/Toronto/Edmonton:")
print(selected_cities.head())

# =========================================================
# 5) تمارين تطبيقية منطقية
# =========================================================

# المثال الأكاديمي 1:
# سعر household heating fuel في Vancouver سنة 1990
vancouver_1990 = data[
    (data['Year'] == 1990) &                         # سنة 1990
    (data['TYPE'] == 'Household heating fuel') &    # نوع الوقود المطلوب
    (data['City'] == 'Vancouver')                   # المدينة المطلوبة
]

print("\nHousehold heating fuel in Vancouver (1990):")
print(vancouver_1990)

# المثال الأكاديمي 2:
# سعر household heating fuel في Vancouver في 1979 و 2021
vancouver_1979_2021 = data[
    (data['Year'].isin([1979, 2021])) &             # اختيار سنتين محددتين
    (data['TYPE'] == 'Household heating fuel') &    # نوع الوقود المطلوب
    (data['City'] == 'Vancouver')                   # المدينة المطلوبة
]

print("\nHousehold heating fuel in Vancouver (1979 and 2021):")
print(vancouver_1979_2021)

# =========================================================
# 6) التجميع باستخدام groupby
# =========================================================

# عدد المجموعات في GEO
geo_groups = data.groupby('GEO')                    # تجميع حسب الموقع الكامل
print("\nعدد مجموعات GEO:")
print(geo_groups.ngroups)                           # عدد المدن/المواقع المختلفة

# متوسط الأسعار لكل سنة
mean_price_by_year = data.groupby('Year')['VALUE'].mean()   # متوسط قيمة الوقود حسب السنة
print("\nمتوسط السعر لكل سنة:")
print(mean_price_by_year.head())

# أعلى سعر لكل شهر
max_price_by_month = data.groupby('Month')['VALUE'].max()   # أقصى قيمة لكل شهر
print("\nأعلى سعر في كل شهر:")
print(max_price_by_month)

# الوسيط لكل سنة ولكل مدينة
median_by_year_city = data.groupby(['Year', 'City'])['VALUE'].median()  # median لكل (سنة، مدينة)
print("\nوسيط السعر لكل سنة ولكل مدينة:")
print(median_by_year_city.head(15))

# =========================================================
# 7) تجهيز البيانات للرسم البياني
# =========================================================

# متوسط السعر السنوي لكل مدينة
price_by_city = (
    data.groupby(['Year', 'GEO'])['VALUE']
    .mean()
    .reset_index(name='AveragePrice')               # إعادة الفهرس وتحويل الناتج لجدول
    .round(2)                                       # تقريب الأرقام إلى منزلتين
)

# =========================================================
# 8) رسم خطي: تطور متوسط الأسعار عبر السنوات لكل مدينة
# =========================================================
fig1 = px.line(
    price_by_city,
    x='Year',
    y='AveragePrice',
    color='GEO',
    title='Average Gasoline Price by City Over Time'   # عنوان الرسم
)

fig1.update_traces(mode='markers+lines')               # إظهار نقاط + خطوط
fig1.update_layout(
    xaxis_title='Year',                                # عنوان المحور الأفقي
    yaxis_title='Average Price',                       # عنوان المحور العمودي
    legend_title='City/Province'                       # عنوان وسيلة الإيضاح
)

fig1.show()                                            # عرض الرسم

# =========================================================
# 9) رسم خطي: متوسط السعر الشهري في Toronto سنة 2021
# =========================================================
toronto_2021 = data[
    (data['Year'] == 2021) &
    (data['GEO'] == 'Toronto, Ontario')
]                                                      # فلترة بيانات تورونتو لسنة 2021 فقط

monthly_toronto = (
    toronto_2021.groupby('Month')['VALUE']
    .mean()
    .reset_index()
)

# ترتيب الأشهر ترتيبًا صحيحًا بدل الترتيب الأبجدي
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']   # ترتيب الأشهر الصحيح
monthly_toronto['Month'] = pd.Categorical(
    monthly_toronto['Month'],
    categories=month_order,
    ordered=True
)
monthly_toronto = monthly_toronto.sort_values('Month')     # ترتيب الأشهر زمنيًا

fig2 = px.line(
    monthly_toronto,
    x='Month',
    y='VALUE',
    title='Toronto Average Monthly Gasoline Price in 2021'
)

fig2.update_traces(mode='markers+lines')                   # نقاط وخطوط معًا
fig2.update_layout(
    xaxis_title='Month',
    yaxis_title='Average Price'
)

fig2.show()

# =========================================================
# 10) رسم خطي: متوسط السعر السنوي لكل نوع وقود
# =========================================================
type_gas = (
    data.groupby(['Year', 'TYPE'])['VALUE']
    .mean()
    .reset_index(name='AveragePrice')
    .round(2)
)

fig3 = px.line(
    type_gas,
    x='Year',
    y='AveragePrice',
    color='TYPE',
    title='Annual Average Gasoline Price by Fuel Type'
)

fig3.update_traces(mode='markers+lines')
fig3.update_layout(
    xaxis_title='Year',
    yaxis_title='Average Price',
    legend_title='Fuel Type'
)

fig3.show()

# =========================================================
# 11) رسم متحرك: تغيّر متوسط السعر حسب المدينة عبر السنوات
# =========================================================
by_city = (
    data.groupby(['Year', 'City'])['VALUE']
    .mean()
    .reset_index(name='AveragePrice')
    .round(2)
)

fig4 = px.bar(
    by_city,
    x='City',
    y='AveragePrice',
    animation_frame='Year',                          # كل سنة تعتبر frame في الحركة
    title='Time Lapse of Average Price by City'
)

fig4.update_layout(
    xaxis_title='City',
    yaxis_title='Average Price'
)

fig4.show()

# =========================================================
# 12) إنشاء خريطة Choropleth لمتوسط السعر في 2021
# =========================================================

# اختيار بيانات سنة 2021 فقط
one_year = data[data['Year'] == 2021]               # فلترة سنة 2021

# حساب متوسط السعر لكل مقاطعة
geodata = (
    one_year.groupby('Province')['VALUE']
    .mean()
    .reset_index(name='Average Gasoline Price')
    .round(2)
)

# ربط كل مقاطعة بمعرف رقمي كما هو مستخدم في ملف الخريطة
provinces = {
    'Newfoundland and Labrador': 5,
    'Prince Edward Island': 8,
    'Nova Scotia': 2,
    'New Brunswick': 7,
    'Quebec': 9,
    'Ontario': 10,
    'Manitoba': 4,
    'Saskatchewan': 1,
    'Alberta': 6,
    'British Columbia': 3
}

geodata['ProvinceID'] = geodata['Province'].map(provinces)   # تحويل اسم المقاطعة إلى رقم تعريف

# جلب ملف الخريطة geojson
geo_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/canada_provinces.geojson"
geo_response = requests.get(geo_url)                # إرسال طلب لجلب ملف الخريطة
map_json = json.loads(geo_response.text)            # تحويل النص إلى JSON مفهوم

# رسم الخريطة
fig5 = px.choropleth(
    geodata,
    locations='ProvinceID',                         # العمود الذي يحتوي معرف المقاطعة
    geojson=map_json,                               # ملف الخريطة
    featureidkey='properties.cartodb_id',           # المفتاح المطابق داخل geojson
    color='Average Gasoline Price',                 # العمود الذي يحدد لون المقاطعة
    hover_name='Province',                          # الاسم الظاهر عند تمرير المؤشر
    color_continuous_scale=px.colors.diverging.Tropic,  # تدرج الألوان
    title='Average Gasoline Price by Province in 2021'
)

fig5.update_geos(fitbounds="locations", visible=False)  # جعل الخريطة تركز تلقائيًا على المواقع
fig5.show()

# =========================================================
# 13) ملاحظات دراسية سريعة
# =========================================================
print("\nملاحظات مهمة:")
print("- استخدم & مع الشروط المتعددة جميعها")
print("- استخدم | عندما تريد (أو)")
print("- استخدم isin() عندما تكون عندك قائمة قيم")
print("- استخدم groupby() للتلخيص الإحصائي")
print("- استخدم reset_index() لتحويل ناتج groupby إلى DataFrame عادي")
print("- الرسوم البيانية تجعل الأنماط الزمنية والمكانية أوضح بكثير")