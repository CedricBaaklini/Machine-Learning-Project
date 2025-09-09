import pandas as pd
import matplotlib.pyplot as plt

#Reading CSV file
df = pd.read_csv('Mall_Customers.csv')

#Calculating Average Annual Income of the customers
average_income = df['Annual Income (k$)'].mean()

print(f"Average annual income of customers: ${average_income}K")

#Showing the number of males vs. females
gender_counts = df['Genre'].value_counts()

plt.figure(figsize = (8, 6))
plt.pie(gender_counts.values, labels = gender_counts.index, autopct = '%1.1f%%', startangle = 90)
plt.title('Number of Male vs. Female Customers')
plt.axis('equal')

plt.show()
