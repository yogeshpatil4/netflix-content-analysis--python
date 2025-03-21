#!/usr/bin/env python
# coding: utf-8

# # 📊 Netflix Content Analysis

# 
# 
# ## 📌 Project Overview
# This project aims to analyze Netflix’s content strategy by examining factors such as content type, language, release season, and timing. By identifying the best-performing content and analyzing release trends, this study provides insights into how Netflix optimizes audience engagement throughout the year.
# 
# 
# 

# ## 🎯 Objective
# The goal is to understand how various factors like content type, language, release season, and timing affect viewership patterns. By analyzing the best-performing content and the timing of its release, we aim to uncover insights into how Netflix maximizes audience engagement throughout the year.
# 

# ## 📂 Dataset Information
# The dataset contains information about Netflix content, including:
# - **Title**: Name of the movie/show.  
# - **Available Globally?**: Indicates if the content is accessible worldwide.  
# - **Release Date**: The date when the content was released.  
# - **Hours Viewed**: Total watch hours (needs cleaning).  
# - **Language Indicator**: The primary language of the content.  
# - **Content Type**: Whether it’s a movie or a show.  

# ## 1️⃣ Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind


# ## 2️⃣ Load and Preview Data
# The dataset used in this analysis can be accessed here:  
# [Netflix Content 2023 Dataset](https://statso.io/netflix-content-strategy-case-study/)
# 

# ### 📂 Load Dataset

# In[2]:


df = pd.read_csv("netflix_content_2023.csv")


# ### 🔍 Preview Data

# In[3]:


# Display first five rows
df.head()


# In[4]:


# Display dataset information
df.info()


# In[5]:


# Summary statistics
df.describe()


# ## 3️⃣ Data Cleaning
# Before analysis, the dataset is cleaned by:

# In[6]:


# Convert 'Hours Viewed' to numeric after removing commas
df['Hours Viewed'] = df['Hours Viewed'].replace(',', '', regex=True).astype(float)

df.head()


# In[7]:


# Convert 'Release Date' to datetime format
df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')


# ## 4️⃣ Exploratory Data Analysis (EDA)

# ### 🔹 Top 10 Most Watched Shows/Movies

# In[8]:


top_10 = df.nlargest(10, 'Hours Viewed')
plt.figure(figsize=(10, 5))
sns.barplot(x='Hours Viewed', y='Title', data=top_10, palette='viridis')
plt.title("Top 10 Most Watched Netflix Content")
plt.xlabel("Total Hours Viewed")
plt.ylabel("Title")
plt.show()


# ### 🔹 Content Type Distribution

# In[9]:


plt.figure(figsize=(10, 5))
sns.countplot(x='Content Type', data=df, palette='coolwarm')
plt.title("Movies vs. Shows on Netflix")
plt.xlabel("Content Type")
plt.ylabel("Count")
plt.show()


# ### 🔹 Aggregate Viewership Hours by Release Month

# In[10]:


# Extract month name from release date
df['Release Month'] = df['Release Date'].dt.strftime('%B')

# Aggregate viewership hours by month and convert to billions
df['Hours Viewed'] = df['Hours Viewed'] / 1e9  # Convert to billions
monthly_viewership = df.groupby('Release Month', observed=False)['Hours Viewed'].sum().reset_index()

# Sort months in calendar order
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
monthly_viewership['Release Month'] = pd.Categorical(monthly_viewership['Release Month'], categories=month_order, ordered=True)
monthly_viewership = monthly_viewership.sort_values('Release Month')

# Plot line chart
plt.figure(figsize=(12, 6))
sns.lineplot(x='Release Month', y='Hours Viewed', data=monthly_viewership, marker='o', color='b')
plt.title("Total Viewership Hours by Release Month (in Billions)")
plt.xlabel("Month")
plt.ylabel("Total Hours Viewed (Billions)")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# ### 🔹 Aggregate Viewership Hours by Content Type and Release Month

# In[11]:


# Aggregate viewership by content type and release month
df_grouped = df.groupby(['Content Type', 'Release Month'], observed=False)['Hours Viewed'].sum().reset_index()

# Sort months in calendar order
df_grouped['Release Month'] = pd.Categorical(df_grouped['Release Month'], categories=month_order, ordered=True)
df_grouped = df_grouped.sort_values(['Content Type', 'Release Month'])

# Plot line chart
plt.figure(figsize=(12, 6))
sns.lineplot(x='Release Month', y='Hours Viewed', hue='Content Type', data=df_grouped, marker='o')
plt.title("Viewership Hours by Content Type and Release Month (in Billions)")
plt.xlabel("Month")
plt.ylabel("Total Hours Viewed (Billions)")
plt.xticks(rotation=45)
plt.legend(title='Content Type')
plt.grid(True)
plt.show()


# ### 💡 Total Viewership Hours by Release Season

# In[12]:


# Define Seasons Based on Release Months
season_mapping = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
}
df['Season'] = df['Release Date'].dt.month.map(season_mapping)


# In[13]:


# Total Viewership Hours by Release Season
seasonal_viewership = df.groupby('Season')['Hours Viewed'].sum().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(x='Season', y='Hours Viewed', data=seasonal_viewership, palette='coolwarm')
plt.title("Total Viewership Hours by Release Season (in Billions)")
plt.xlabel("Season")
plt.ylabel("Total Hours Viewed (Billions)")
plt.show()


# ### 💡 Aggregate Viewership Hours by Language

# In[14]:


language_viewership = df.groupby('Language Indicator')['Hours Viewed'].sum().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='Hours Viewed', y='Language Indicator', data=language_viewership, palette='viridis')
plt.title("Total Viewership Hours by Language (in Billions)")
plt.xlabel("Total Hours Viewed (Billions)")
plt.ylabel("Language")
plt.show()


# ### 💡 Monthly Release Patterns and Viewership Hours
# 

# In[15]:


df['Release Month'] = df['Release Date'].dt.strftime('%B')
df['Hours Viewed'] = df['Hours Viewed'] / 1e9  # Convert to billions
monthly_data = df.groupby('Release Month', observed=False).agg(
    Release_Count=('Title', 'count'),
    Hours_Viewed=('Hours Viewed', 'sum')
).reset_index()

# Sort months in calendar order
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
monthly_data['Release Month'] = pd.Categorical(monthly_data['Release Month'], categories=month_order, ordered=True)
monthly_data = monthly_data.sort_values('Release Month')

# Plot Monthly Data
fig, ax1 = plt.subplots(figsize=(14, 6))
sns.barplot(x='Release Month', y='Release_Count', data=monthly_data, ax=ax1, color='lightblue', alpha=0.6, label='No. of Releases')
ax2 = ax1.twinx()
sns.lineplot(x='Release Month', y='Hours_Viewed', data=monthly_data, ax=ax2, marker='o', color='red', label='Viewership Hours')
ax1.set_xlabel("Month")
ax1.set_ylabel("Number of Releases", color='blue')
ax2.set_ylabel("Total Hours Viewed (Billions)", color='red')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title("Monthly Release Patterns and Viewership Hours")
plt.xticks(rotation=45)
plt.show()


# ### 💡 Weekly Release Patterns and Viewership Hours

# In[16]:


df['Release Day'] = df['Release Date'].dt.strftime('%A')
weekly_data = df.groupby('Release Day', observed=False).agg(
    Release_Count=('Title', 'count'),
    Hours_Viewed=('Hours Viewed', 'sum')
).reset_index()

# Sort days in order
week_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_data['Release Day'] = pd.Categorical(weekly_data['Release Day'], categories=week_order, ordered=True)
weekly_data = weekly_data.sort_values('Release Day')

# Plot Weekly Data
fig, ax1 = plt.subplots(figsize=(12, 6))
sns.barplot(x='Release Day', y='Release_Count', data=weekly_data, ax=ax1, color='lightgreen', alpha=0.6, label='No. of Releases')
ax2 = ax1.twinx()
sns.lineplot(x='Release Day', y='Hours_Viewed', data=weekly_data, ax=ax2, marker='o', color='purple', label='Viewership Hours')
ax1.set_xlabel("Day of the Week")
ax1.set_ylabel("Number of Releases", color='green')
ax2.set_ylabel("Total Hours Viewed (Billions)", color='purple')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title("Weekly Release Patterns and Viewership Hours")
plt.xticks(rotation=45)
plt.show()


# ## 5️⃣ Statistical Analysis

# ### 💡Correlation Analysis
# Is there a relationship between the release year and viewing hours?

# In[17]:


# Calculate Pearson correlation
correlation = df['Release Date'].dt.year.corr(df['Hours Viewed'])
print(f"Pearson Correlation: {correlation}")


# #### Insights
# - The correlation between `Release Year` and `Hours Viewed` is **0.19**, indicating a weak positive relationship. Newer releases tend to have slightly higher viewing hours.
# - The line plot shows that total viewing hours have increased over the years, with a significant spike in **2023**.

# ### 💡 Hypothesis Testing
# We will perform a two-sample t-test to compare the average hours viewed between two groups:
# 
# 1. English Content
# 2. Non-English Content
# 
# Hypotheses:
# 
# **Null Hypothesis (H₀):** There is no difference in the average hours viewed between English and non-English content.
# 
# **Alternative Hypothesis (H₁):** There is a significant difference in the average hours viewed between English and non-English content.

# In[18]:


# Split the data into English and non-English content
english_content = df[df['Language Indicator'] == 'English']
non_english_content = df[df['Language Indicator'] != 'English']

# Perform the two-sample t-test
t_stat, p_value = ttest_ind(english_content['Hours Viewed'], non_english_content['Hours Viewed'], equal_var=False)

# Output the results
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")


# #### Result: 
# The t-statistic of *10.42* indicates a large difference between the means of the two groups (English and non-English content). The p-value of *2.25* is greater than the typical significance level of *0.05*, which means **we fail to reject the null hypothesis**.
# 
# #### Insight: 
# There is no statistically significant difference in the average hours viewed between English and non-English content. This suggests that both English and non-English content perform similarly in terms of viewership on Netflix.

# ## 6️⃣ Key Insights from Analysis  
# 
# ### **1 Viewership Trends & Popular Content**  
# ✅ A few shows and movies drive a significant share of total watch hours.  
# ✅ Movies tend to attract slightly higher viewership than shows.  
# 
# ### **2 Seasonal & Monthly Performance**  
# ✅ **Viewership peaks in winter (December - February)** due to holidays.  
# ✅ The highest number of content releases does not always align with peak viewership.  
# 
# ### **3 Language & Global Impact**  
# ✅ English and non-English content perform similarly, proving that Netflix's international expansion is effective.  
# 
# ### **4 Correlation & Statistical Analysis**  
# ✅ Weak positive correlation (0.19) between release year and viewership hours.  
# ✅ **T-test shows no significant difference** in average hours viewed between English and non-English content.  
# 
# 
# 
# 

# ## 7️⃣ Conclusion  
# Netflix's content strategy is **data-driven and global-focused**. Based on the analysis, key recommendations include:  
# 🔹 **Optimizing release schedules** by focusing on high-viewership seasons.  
# 🔹 **Strengthening investment in non-English content** to expand its global market.  
# 🔹 **Further analyzing genre preferences** to refine content recommendations.  
# 
# ---
# 
# *This analysis provides actionable insights to enhance Netflix's content strategy and maximize audience engagement.* 🎬📈

# In[ ]:




