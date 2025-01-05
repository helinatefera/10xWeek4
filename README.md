# Rossmann Pharmaceuticals: Sales Forecasting and Analysis  

## Project Overview  
This project focuses on building and serving an end-to-end product to forecast sales for Rossmann Pharmaceuticals stores across several cities. The goal is to provide accurate six-week sales predictions, empowering the finance team and store managers with data-driven insights.  

The project incorporates analysis of factors such as:  
- **Promotions**  
- **Competitor Activity**  
- **Weekend and State Holidays**  

By integrating machine learning techniques and exploratory data analysis, we offer actionable insights into sales dynamics and influential factors.  

---

## Features  

- **Promotion Analysis**: Evaluate the effectiveness of promotional campaigns.  
- **Competitor Analysis**: Assess the impact of competitor proximity and openings on store performance.  
- **Customer Behavior Trends**: Explore customer activity during weekends, holidays, and store operational hours.  
- **Seasonality Insights**: Identify patterns in sales based on weekend.  

---

## Data Description  

The dataset contains the following columns:  
- **Date**: Daily data for each store.  
- **Store**: Unique ID for each store.  
- **DayOfWeek**: The day of the week.  
- **Sales**: Sales for the store on the given day.  
- **Customers**: Number of customers for the store on the given day.  
- **Promo**: Whether a store is running a promotion that day.  
- **CompetitionDistance**: Distance to the nearest competitor store.  
- **Assortment**: Assortment type ('a', 'b', 'c').  
- **Promo2**: Whether the store is part of a continuous promotion.  

---

## Methodology  

1. **Exploratory Data Analysis (EDA)**:  
   - Analyzed sales trends based on store type, promotions, and competitor activity.  
   - Visualized customer behavior trends during weekends and holidays.  

2. **Data Preprocessing**:  
   - Merged multiple datasets (store details, sales data, promotional data).  
   - Handled missing values and ensured consistency in time-series data.  

3. **Promotion Analysis**:  
   - Evaluated the sales uplift during promotional periods.  
   - Assessed the effectiveness of promotions on customer engagement.  

4. **Competitor Impact Analysis**:  
   - Correlated sales with the proximity and opening of competitors.  
   - Investigated the impact of competition on store performance.    

---

## Key Results  

- **Promotion Effectiveness**: Promotions significantly boosted sales, particularly in stores with extended assortments.  
- **Competitor Impact**: The presence of nearby competitors slightly reduced sales, though city-center stores were less affected.  
- **Customer Trends**: Weekend sales were normal like other days, with promotions. 

---

## Installation  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/helinatefera/10xWeek4.git  
   ```  

2. Navigate to the project directory:  
   ```bash  
   cd 10xWeek4  
   ```  


---

## Usage  
 
1. **Promotion Insights**: Refer to the `notebooks/tasx_1.ipynb` notebook for insights into promotional effectiveness.  
2. **Competitor Impact**: Explore the impact of competitor activity using the `notebooks/task_1.ipynb` notebook.  

---

## Contributing  

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.  

---

## License  

This project is licensed under the [MIT License](LICENSE).  
