Original file is located at
    https://colab.research.google.com/drive/15TPe5z8uYG1Erw0YxPAVfsIwvExgBWNn

# Zomato Restaurant and Review Data Analysis

Goal: Analyze restaurant and review data to extract insights on ratings, cuisines, location preferences and factors affecting ratings.

## 1. Import Required Libraries
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# To show charts clearly
plt.rcParams['figure.figsize'] = (10, 5)

"""## 2. Load Dataset"""

# Upload zomato.csv / zomato(1).csv in Colab Files section before running

file_name = '/content/zomato.csv'   # change to 'zomato.csv' if your file name is zomato.csv
df = pd.read_csv(file_name)

df.head()

"""## 3. Basic Dataset Check"""

print('Rows and Columns:', df.shape)
print('\nColumn Names:')
print(df.columns)

df.info()

"""## 4. Data Cleaning
Handle duplicate rows, missing values, rating text, and cost currency format.
"""

# Check missing values
df.isnull().sum()

# Remove duplicate rows
df = df.drop_duplicates()

# Clean rating column: convert values like '4.1/5' into 4.1
df['rate'] = df['rate'].astype(str).str.replace('/5', '', regex=False).str.strip()
df['rate'] = df['rate'].replace(['NEW', '-', 'nan'], pd.NA)
df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

# Clean cost column: convert values like '1,200' into 1200
df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str).str.replace(',', '', regex=False).str.strip()
df['approx_cost(for two people)'] = df['approx_cost(for two people)'].replace('nan', pd.NA)
df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')

# Clean votes column
df['votes'] = pd.to_numeric(df['votes'], errors='coerce')

# Fill important text missing values
for col in ['location', 'cuisines', 'rest_type', 'listed_in(type)']:
    df[col] = df[col].fillna('Unknown')

# Remove rows where important numeric values are missing
df = df.dropna(subset=['rate', 'approx_cost(for two people)', 'votes'])

print('Cleaned dataset shape:', df.shape)
df.head()

"""## 5. Cuisine vs Rating
Find cuisines with the highest average ratings.
"""

cuisine_rating = df.groupby('cuisines')['rate'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 5))
cuisine_rating.plot(kind='bar')
plt.title('Top 10 Cuisines by Average Rating')
plt.xlabel('Cuisines')
plt.ylabel('Average Rating')
plt.xticks(rotation=75)
plt.show()

"""## 6. Location Hotspots
Find locations with the highest number of restaurants.
"""

location_hotspots = df['location'].value_counts().head(10)

plt.figure(figsize=(12, 5))
location_hotspots.plot(kind='bar')
plt.title('Top 10 Restaurant Location Hotspots')
plt.xlabel('Location')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=75)
plt.show()

"""## 7. Price vs Rating
Check the relationship between cost for two people and rating.
"""

plt.figure(figsize=(10, 5))
sns.scatterplot(x='approx_cost(for two people)', y='rate', data=df)
plt.title('Price vs Rating')
plt.xlabel('Approx Cost for Two People')
plt.ylabel('Rating')
plt.show()

"""## 8. Heatmap
Show relationship between rating, votes, and cost.
"""

numeric_cols = df[['rate', 'votes', 'approx_cost(for two people)']]

plt.figure(figsize=(7, 5))
sns.heatmap(numeric_cols.corr(), annot=True)
plt.title('Correlation Heatmap')
plt.show()

"""## 9. WordCloud for Popular Cuisines"""

text = ' '.join(df['cuisines'].dropna().astype(str))

wordcloud = WordCloud(width=900, height=450, background_color='white').generate(text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud of Popular Cuisines')
plt.show()

"""## 10. Key Insights

1. Restaurant concentration is higher in a few major locations, showing clear location hotspots.
2. Some cuisines have better average ratings than others, which helps identify customer-preferred cuisine categories.
3. Price and rating do not always increase together; expensive restaurants are not automatically higher rated.
4. Votes are useful because restaurants with more votes usually give more reliable rating signals.
5. Cuisine and location together strongly influence restaurant visibility and customer preference.

## 11. Recommendations for Alfido Tech Style Platform

1. Partner with high-rated restaurants from top-performing cuisine categories.
2. Promote restaurants located in major hotspot areas to increase user engagement.
3. Create cuisine-based recommendation sections such as North Indian, Chinese, Cafe, and Fast Food.
4. Use both rating and votes for restaurant ranking, not rating alone.
5. Provide budget-based filters because price does not always guarantee better rating.

## 12. Save Cleaned Dataset
"""

df.to_csv('zomato_cleaned.csv', index=False)
print('Cleaned dataset saved as zomato_cleaned.csv')
