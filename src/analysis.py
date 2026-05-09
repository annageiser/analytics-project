# ================================================================================================
# ===================================   Exploratory Data Analysis   ==============================
# ================================================================================================

# Load libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load data
train = pd.read_csv('../data/raw/train.csv', sep='|')
items = pd.read_csv('../data/raw/items.csv', sep='|')

df = train.merge(items, 
                   on='pid', 
                   how='left',
                   validate="m:1")

# -------------------------------------------------------------------------------------------------

# Initial Data Analysis
pd.set_option('display.max_columns', None)
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.nunique())


# Correlation Analysis
print(df.corr(numeric_only=True))
df.corr(numeric_only=True).to_csv('../reports/correlation_matrix.csv')

# Visualization of correlation matrix
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig('../reports/figures/correlation_matrix.png')
plt.show()

print(df.corr(numeric_only=True)['order'].sort_values(ascending=False))

# -------------------------------------------------------------------------------------------------

# Further Data Analysis

# Target Variable

# Order

# Numbers
order_counts = df['order'].value_counts()
order_percentages = (df['order'].value_counts(normalize=True) * 100).round(2)
order_EDA = pd.DataFrame({'count': order_counts, 'percentage': order_percentages})
order_EDA.to_csv('../data/processed/order_EDA.csv', index=True)
print(order_EDA)

# Visualization
plot = sns.countplot(x="order", hue="order", data=df)
if plot.legend_ is not None:
    plot.legend_.remove()
plot.get_yaxis().get_major_formatter().set_scientific(False)
plt.title("Distribution of Order Flag")
plt.savefig('../reports/figures/order_EDA.png')
plt.show()


# Funnel Analysis

# Basket

# Numbers
basket_counts = df['basket'].value_counts()
basket_percentages = (df['basket'].value_counts(normalize=True) * 100).round(2)
basket_order_ct = pd.crosstab(df['basket'], df['order'])
basket_order_ct_pct = (pd.crosstab(df['basket'], df['order'], normalize='index') * 100).round(2)
basket_EDA_df = pd.DataFrame({
    'total_count': basket_counts,
    'total_percentage': basket_percentages,
    'order_0_count': basket_order_ct[0],
    'order_1_count': basket_order_ct[1],
    'order_0_percentage': basket_order_ct_pct[0],
    'order_1_percentage': basket_order_ct_pct[1]
})
print(basket_EDA_df)
basket_EDA_df.to_csv('../data/processed/basket_EDA.csv', index=True)

# Visualization
plot = sns.countplot(x="basket", hue="order", data=df)
plot.get_yaxis().get_major_formatter().set_scientific(False)
plt.title("Distribution of Basket Flag")
plt.savefig('../reports/figures/basket_EDA.png')
plt.show()


# Click

# Numbers
click_counts = df['click'].value_counts()
click_percentages = (df['click'].value_counts(normalize=True) * 100).round(2)
click_order_ct = pd.crosstab(df['click'], df['order'])
click_order_ct_pct = (pd.crosstab(df['click'], df['order'], normalize='index') * 100).round(2)
click_EDA_df = pd.DataFrame({
    'total_count': click_counts,
    'total_percentage': click_percentages,
    'order_0_count': click_order_ct[0],
    'order_1_count': click_order_ct[1],
    'order_0_percentage': click_order_ct_pct[0],
    'order_1_percentage': click_order_ct_pct[1]
})
print(click_EDA_df)
click_EDA_df.to_csv('../data/processed/click_EDA.csv', index=True)

# Visualization
plot = sns.countplot(x="click", hue="order", data=df)
plot.get_yaxis().get_major_formatter().set_scientific(False)
plt.title("Distribution of Click Flag")
plt.savefig('../reports/figures/click_EDA.png')
plt.show()


# Categorical Features


# Pid


# - High-Cardinality Categorical Feature
# - Long-tail distribution


# Numbers
all_pid_counts_total = df['pid'].value_counts()
top_20_pids_list = all_pid_counts_total.head(20).index
all_pid_order_counts = df.pivot_table(index='pid', columns='order', aggfunc='size', fill_value=0)
all_pid_order_counts['total_events'] = all_pid_order_counts.sum(axis=1)
top_20_counts = all_pid_order_counts.loc[top_20_pids_list]
other_pids = all_pid_order_counts.drop(top_20_pids_list)
other_counts = other_pids.sum().to_frame('others').T
pid_EDA_df = pd.concat([top_20_counts, other_counts])
total_events = df.shape[0]
total_orders = df['order'].sum()
total_no_orders = total_events - total_orders
pid_EDA_df['total_event_percentage'] = (pid_EDA_df['total_events'] / total_events * 100).round(2)
pid_EDA_df['order_count'] = pid_EDA_df[1]
pid_EDA_df['order_percentage'] = (pid_EDA_df[1] / total_orders * 100).round(2)
pid_EDA_df['no_order_count'] = pid_EDA_df[0]
pid_EDA_df['no_order_percentage'] = (pid_EDA_df[0] / total_no_orders * 100).round(2)
pid_EDA_df = pid_EDA_df[['total_events', 'total_event_percentage', 'order_count', 'order_percentage', 'no_order_count', 'no_order_percentage']]
pid_EDA_df.rename(columns={'total_events': 'total_event_count'}, inplace=True)
pid_EDA_df.to_csv('../data/processed/pid_EDA.csv', index=True)
print("Number of unique PIDs:", df['pid'].unique().shape[0])
print("\nTop 20 PIDs (and Others) by Interaction and Order Count:")
print(pid_EDA_df)

# Visualization
top_20_pids_list_viz = df['pid'].value_counts().head(20).index
df_top20_pid = df[df['pid'].isin(top_20_pids_list_viz)]
pid_order_counts_viz = df_top20_pid.pivot_table(index='pid', columns='order', aggfunc='size', fill_value=0)
pid_order_counts_viz = pid_order_counts_viz.reindex(top_20_pids_list_viz)
pid_order_counts_viz = pid_order_counts_viz[[1, 0]]
colors = ['tab:orange', 'tab:blue']
pid_order_counts_viz.plot(kind='barh', stacked=True, figsize=(12, 8), color=colors)
plt.title('Distribution of Top 20 PIDs by Order Status')
plt.xlabel('Count')
plt.ylabel('PID')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../reports/figures/pid_EDA.png')
plt.show()


# #### ***Manufacturer***


# - High-Cardinality Categorical Feature
# - Long-tail distribution


# Numbers
all_manufacturer_counts_total = df['manufacturer'].value_counts()
top_20_manufacturer_list = all_manufacturer_counts_total.head(20).index
all_manufacturer_order_counts = df.pivot_table(index='manufacturer', columns='order', aggfunc='size', fill_value=0)
all_manufacturer_order_counts['total_events'] = all_manufacturer_order_counts.sum(axis=1)
top_20_counts = all_manufacturer_order_counts.loc[top_20_manufacturer_list]
other_manufacturers = all_manufacturer_order_counts.drop(top_20_manufacturer_list)
other_counts = other_manufacturers.sum().to_frame('others').T
manufacturer_EDA_df = pd.concat([top_20_counts, other_counts])
total_events = df.shape[0]
total_orders = df['order'].sum()
total_no_orders = total_events - total_orders
manufacturer_EDA_df['total_event_percentage'] = (manufacturer_EDA_df['total_events'] / total_events * 100).round(2)
manufacturer_EDA_df['order_count'] = manufacturer_EDA_df[1]
manufacturer_EDA_df['order_percentage'] = (manufacturer_EDA_df[1] / total_orders * 100).round(2)
manufacturer_EDA_df['no_order_count'] = manufacturer_EDA_df[0]
manufacturer_EDA_df['no_order_percentage'] = (manufacturer_EDA_df[0] / total_no_orders * 100).round(2)
manufacturer_EDA_df = manufacturer_EDA_df[['total_events', 'total_event_percentage', 'order_count', 'order_percentage', 'no_order_count', 'no_order_percentage']]
manufacturer_EDA_df.rename(columns={'total_events': 'total_event_count'}, inplace=True)
manufacturer_EDA_df.to_csv('../data/processed/manufacturer_EDA.csv', index=True)
print("Number of unique Manufacturers:", df['manufacturer'].unique().shape[0])
print("\nTop 20 Manufacturers (and Others) by Interaction and Order Count:")
print(manufacturer_EDA_df)

# Visualization
top_20_manufacturers_list_viz = df['manufacturer'].value_counts().head(20).index
df_top20_manufacturer = df[df['manufacturer'].isin(top_20_manufacturers_list_viz)]
manufacturer_order_counts_viz = df_top20_manufacturer.pivot_table(index='manufacturer', columns='order', aggfunc='size', fill_value=0)
manufacturer_order_counts_viz = manufacturer_order_counts_viz.reindex(top_20_manufacturers_list_viz)
manufacturer_order_counts_viz = manufacturer_order_counts_viz[[1, 0]]
colors = ['tab:orange', 'tab:blue']
manufacturer_order_counts_viz.plot(kind='barh', stacked=True, figsize=(12, 8), color=colors)
plt.title('Distribution of Top 20 Manufacturers by Order Status')
plt.xlabel('Count')
plt.ylabel('Manufacturer')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../reports/figures/manufacturer_EDA.png')
plt.show()


# #### ***Content***


# - High-Cardinality Categorical Feature
# - Long-tail distribution


# Numbers
all_content_counts_total = df['content'].value_counts()
top_20_content_list = all_content_counts_total.head(20).index
all_content_order_counts = df.pivot_table(index='content', columns='order', aggfunc='size', fill_value=0)
all_content_order_counts['total_events'] = all_content_order_counts.sum(axis=1)
top_20_counts = all_content_order_counts.loc[top_20_content_list]
other_contents = all_content_order_counts.drop(top_20_content_list)
other_counts = other_contents.sum().to_frame('others').T
content_EDA_df = pd.concat([top_20_counts, other_counts])
total_events = df.shape[0]
total_orders = df['order'].sum()
total_no_orders = total_events - total_orders
content_EDA_df['total_event_percentage'] = (content_EDA_df['total_events'] / total_events * 100).round(2)
content_EDA_df['order_count'] = content_EDA_df[1]
content_EDA_df['order_percentage'] = (content_EDA_df[1] / total_orders * 100).round(2)
content_EDA_df['no_order_count'] = content_EDA_df[0]
content_EDA_df['no_order_percentage'] = (content_EDA_df[0] / total_no_orders * 100).round(2)
content_EDA_df = content_EDA_df[['total_events', 'total_event_percentage', 'order_count', 'order_percentage', 'no_order_count', 'no_order_percentage']]
content_EDA_df.rename(columns={'total_events': 'total_event_count'}, inplace=True)
content_EDA_df.to_csv('../data/processed/content_EDA.csv', index=True)
print("Number of unique Contents:", df['content'].unique().shape[0])
print("\nTop 20 Contents (and Others) by Interaction and Order Count:")
print(content_EDA_df)

# Visualization
top_20_contents_list_viz = df['content'].value_counts().head(20).index
df_top20_content = df[df['content'].isin(top_20_contents_list_viz)]
content_order_counts_viz = df_top20_content.pivot_table(index='content', columns='order', aggfunc='size', fill_value=0)
content_order_counts_viz = content_order_counts_viz.reindex(top_20_contents_list_viz)
content_order_counts_viz = content_order_counts_viz[[1, 0]]
colors = ['tab:orange', 'tab:blue']
content_order_counts_viz.plot(kind='barh', stacked=True, figsize=(12, 8), color=colors)
plt.title('Distribution of Top 20 Contents by Order Status')
plt.xlabel('Count')
plt.ylabel('Content')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../reports/figures/content_EDA.png')
plt.show()


# #### ***Group***


# - High-Cardinality Categorical Feature
# - Long-tail distribution


# Numbers
all_group_counts_total = df['group'].value_counts()
top_20_group_list = all_group_counts_total.head(20).index
all_group_order_counts = df.pivot_table(index='group', columns='order', aggfunc='size', fill_value=0)
all_group_order_counts['total_events'] = all_group_order_counts.sum(axis=1)
top_20_counts = all_group_order_counts.loc[top_20_group_list]
other_groups = all_group_order_counts.drop(top_20_group_list)
other_counts = other_groups.sum().to_frame('others').T
group_EDA_df = pd.concat([top_20_counts, other_counts])
total_events = df.shape[0]
total_orders = df['order'].sum()
total_no_orders = total_events - total_orders
group_EDA_df['total_event_percentage'] = (group_EDA_df['total_events'] / total_events * 100).round(2)
group_EDA_df['order_count'] = group_EDA_df[1]
group_EDA_df['order_percentage'] = (group_EDA_df[1] / total_orders * 100).round(2)
group_EDA_df['no_order_count'] = group_EDA_df[0]
group_EDA_df['no_order_percentage'] = (group_EDA_df[0] / total_no_orders * 100).round(2)
group_EDA_df = group_EDA_df[['total_events', 'total_event_percentage', 'order_count', 'order_percentage', 'no_order_count', 'no_order_percentage']]
group_EDA_df.rename(columns={'total_events': 'total_event_count'}, inplace=True)
group_EDA_df.to_csv('../data/processed/group_EDA.csv', index=True)
print("Number of unique Groups:", df['group'].unique().shape[0])
print("\nTop 20 Groups (and Others) by Interaction and Order Count:")
print(group_EDA_df)

# Visualization
top_20_groups_list_viz = df['group'].value_counts().head(20).index
df_top20_group = df[df['group'].isin(top_20_groups_list_viz)]
group_order_counts_viz = df_top20_group.pivot_table(index='group', columns='order', aggfunc='size', fill_value=0)
group_order_counts_viz = group_order_counts_viz.reindex(top_20_groups_list_viz)
group_order_counts_viz = group_order_counts_viz[[1, 0]]
colors = ['tab:orange', 'tab:blue']
group_order_counts_viz.plot(kind='barh', stacked=True, figsize=(12, 8), color=colors)
plt.title('Distribution of Top 20 Groups by Order Status')
plt.xlabel('Count')
plt.ylabel('Group')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../reports/figures/group_EDA.png')
plt.show()


# #### ***Pharm Form***


# - High-Cardinality Categorical Feature
# - Long-tail distribution


# Numbers
all_pharmForm_counts_total = df['pharmForm'].value_counts()
top_20_pharmForm_list = all_pharmForm_counts_total.head(20).index
all_pharmForm_order_counts = df.pivot_table(index='pharmForm', columns='order', aggfunc='size', fill_value=0)
all_pharmForm_order_counts['total_events'] = all_pharmForm_order_counts.sum(axis=1)
top_20_counts = all_pharmForm_order_counts.loc[top_20_pharmForm_list]
other_pharmForms = all_pharmForm_order_counts.drop(top_20_pharmForm_list)
other_counts = other_pharmForms.sum().to_frame('others').T
pharmForm_EDA_df = pd.concat([top_20_counts, other_counts])
total_events = df.shape[0]
total_orders = df['order'].sum()
total_no_orders = total_events - total_orders
pharmForm_EDA_df['total_event_percentage'] = (pharmForm_EDA_df['total_events'] / total_events * 100).round(2)
pharmForm_EDA_df['order_count'] = pharmForm_EDA_df[1]
pharmForm_EDA_df['order_percentage'] = (pharmForm_EDA_df[1] / total_orders * 100).round(2)
pharmForm_EDA_df['no_order_count'] = pharmForm_EDA_df[0]
pharmForm_EDA_df['no_order_percentage'] = (pharmForm_EDA_df[0] / total_no_orders * 100).round(2)
pharmForm_EDA_df = pharmForm_EDA_df[['total_events', 'total_event_percentage', 'order_count', 'order_percentage', 'no_order_count', 'no_order_percentage']]
pharmForm_EDA_df.rename(columns={'total_events': 'total_event_count'}, inplace=True)
pharmForm_EDA_df.to_csv('../data/processed/pharmForm_EDA.csv', index=True)
print("Number of unique PharmForms:", df['pharmForm'].unique().shape[0])
print("\nTop 20 PharmForms (and Others) by Interaction and Order Count:")
print(pharmForm_EDA_df)

# Visualization
top_20_pharmForms_list_viz = df['pharmForm'].value_counts().head(20).index
df_top20_pharmForm = df[df['pharmForm'].isin(top_20_pharmForms_list_viz)]
pharmForm_order_counts_viz = df_top20_pharmForm.pivot_table(index='pharmForm', columns='order', aggfunc='size', fill_value=0)
pharmForm_order_counts_viz = pharmForm_order_counts_viz.reindex(top_20_pharmForms_list_viz)
pharmForm_order_counts_viz = pharmForm_order_counts_viz[[1, 0]]
colors = ['tab:orange', 'tab:blue']
pharmForm_order_counts_viz.plot(kind='barh', stacked=True, figsize=(12, 8), color=colors)
plt.title('Distribution of Top 20 PharmForms by Order Status')
plt.xlabel('Count')
plt.ylabel('PharmForm')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../reports/figures/pharmForm_EDA.png')
plt.show()


# #### ***Category***


# - High-Cardinality Categorical Feature
# - Long-tail distribution


# Numbers
all_category_counts_total = df['category'].value_counts()
top_20_category_list = all_category_counts_total.head(20).index
all_category_order_counts = df.pivot_table(index='category', columns='order', aggfunc='size', fill_value=0)
all_category_order_counts['total_events'] = all_category_order_counts.sum(axis=1)
top_20_counts = all_category_order_counts.loc[top_20_category_list]
other_categories = all_category_order_counts.drop(top_20_category_list)
other_counts = other_categories.sum().to_frame('others').T
category_EDA_df = pd.concat([top_20_counts, other_counts])
total_events = df.shape[0]
total_orders = df['order'].sum()
total_no_orders = total_events - total_orders
category_EDA_df['total_event_percentage'] = (category_EDA_df['total_events'] / total_events * 100).round(2)
category_EDA_df['order_count'] = category_EDA_df[1]
category_EDA_df['order_percentage'] = (category_EDA_df[1] / total_orders * 100).round(2)
category_EDA_df['no_order_count'] = category_EDA_df[0]
category_EDA_df['no_order_percentage'] = (category_EDA_df[0] / total_no_orders * 100).round(2)
category_EDA_df = category_EDA_df[['total_events', 'total_event_percentage', 'order_count', 'order_percentage', 'no_order_count', 'no_order_percentage']]
category_EDA_df.rename(columns={'total_events': 'total_event_count'}, inplace=True)
category_EDA_df.to_csv('../data/processed/category_EDA.csv', index=True)
print("Number of unique Categories:", df['category'].unique().shape[0])
print("\nTop 20 Categories (and Others) by Interaction and Order Count:")
print(category_EDA_df)

# Visualization
top_20_categories_list_viz = df['category'].value_counts().head(20).index
df_top20_category = df[df['category'].isin(top_20_categories_list_viz)]
category_order_counts_viz = df_top20_category.pivot_table(index='category', columns='order', aggfunc='size', fill_value=0)
category_order_counts_viz = category_order_counts_viz.reindex(top_20_categories_list_viz)
category_order_counts_viz = category_order_counts_viz[[1, 0]]
colors = ['tab:orange', 'tab:blue']
category_order_counts_viz.plot(kind='barh', stacked=True, figsize=(12, 8), color=colors)
plt.title('Distribution of Top 20 Categories by Order Status')
plt.xlabel('Count')
plt.ylabel('Category')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../reports/figures/category_EDA.png')
plt.show()


# #### ***Day***


# Numbers
day_counts = df['day'].value_counts()
day_percentages = (df['day'].value_counts(normalize=True) * 100).round(2)
day_order_ct = pd.crosstab(df['day'], df['order'])
day_order_ct_pct = (pd.crosstab(df['day'], df['order'], normalize='index') * 100).round(2)
day_EDA_df = pd.DataFrame({
    'total_count': day_counts,
    'total_percentage': day_percentages,
    'order_0_count': day_order_ct[0],
    'order_1_count': day_order_ct[1],
    'order_0_percentage': day_order_ct_pct[0],
    'order_1_percentage': day_order_ct_pct[1]
})
print(day_EDA_df)
day_EDA_df.to_csv('../data/processed/day_EDA.csv', index=True)

# Visualization
plt.figure(figsize=(20, 4))
plot = sns.countplot(data=df, x="day", hue="order")
plot.set_title("Distribution of Day")
plot.tick_params(axis="x", labelrotation=45)
for label in plot.get_xticklabels():
    label.set_ha("right")
plt.tight_layout()
plt.savefig('../reports/figures/day_EDA.png')
plt.show()


# #### ***Ad Flag***


# Numbers
adFlag_counts = df['adFlag'].value_counts()
adFlag_percentages = (df['adFlag'].value_counts(normalize=True) * 100).round(2)
adFlag_order_ct = pd.crosstab(df['adFlag'], df['order'])
adFlag_order_ct_pct = (pd.crosstab(df['adFlag'], df['order'], normalize='index') * 100).round(2)
adFlag_EDA_df = pd.DataFrame({
    'total_count': adFlag_counts,
    'total_percentage': adFlag_percentages,
    'order_0_count': adFlag_order_ct[0],
    'order_1_count': adFlag_order_ct[1],
    'order_0_percentage': adFlag_order_ct_pct[0],
    'order_1_percentage': adFlag_order_ct_pct[1]
})
print(adFlag_EDA_df)
adFlag_EDA_df.to_csv('../data/processed/adFlag_EDA.csv', index=True)

# Visualization
plot = sns.countplot(x="adFlag", hue="order", data=df)
plot.get_yaxis().get_major_formatter().set_scientific(False)
plt.title("Distribution of Ad Flag")
plt.savefig('../reports/figures/adFlag_EDA.png')
plt.show()


# #### ***Availability***


# Numbers
availability_counts = df['availability'].value_counts()
availability_percentages = (df['availability'].value_counts(normalize=True) * 100).round(2)
availability_order_ct = pd.crosstab(df['availability'], df['order'])
availability_order_ct_pct = (pd.crosstab(df['availability'], df['order'], normalize='index') * 100).round(2)
availability_EDA_df = pd.DataFrame({
    'total_count': availability_counts,
    'total_percentage': availability_percentages,
    'order_0_count': availability_order_ct[0],
    'order_1_count': availability_order_ct[1],
    'order_0_percentage': availability_order_ct_pct[0],
    'order_1_percentage': availability_order_ct_pct[1]
})
print(availability_EDA_df)
availability_EDA_df.to_csv('../data/processed/availability_EDA.csv', index=True)

# Visualization
plot = sns.countplot(x="availability", hue="order", data=df)
plot.get_yaxis().get_major_formatter().set_scientific(False)
plt.title("Distribution of Availability Flag")
plt.savefig('../reports/figures/availability_EDA.png')
plt.show()


# #### ***Unit***


# Numbers
unit_counts = df['unit'].value_counts()
unit_percentages = (df['unit'].value_counts(normalize=True) * 100).round(2)
unit_order_ct = pd.crosstab(df['unit'], df['order'])
unit_order_ct_pct = (pd.crosstab(df['unit'], df['order'], normalize='index') * 100).round(2)
unit_EDA_df = pd.DataFrame({
    'total_count': unit_counts,
    'total_percentage': unit_percentages,
    'order_0_count': unit_order_ct[0],
    'order_1_count': unit_order_ct[1],
    'order_0_percentage': unit_order_ct_pct[0],
    'order_1_percentage': unit_order_ct_pct[1]
})
print(unit_EDA_df)
unit_EDA_df.to_csv('../data/processed/unit_EDA.csv', index=True)

# Visualization
plot = sns.countplot(x="unit", hue="order", data=df)
plot.get_yaxis().get_major_formatter().set_scientific(False)
plt.title("Distribution of Unit Flag")
plt.savefig('../reports/figures/unit_EDA.png')
plt.show()


# #### ***Generic Product***


# Numbers
genericProduct_counts = df['genericProduct'].value_counts()
genericProduct_percentages = (df['genericProduct'].value_counts(normalize=True) * 100).round(2)
genericProduct_order_ct = pd.crosstab(df['genericProduct'], df['order'])
genericProduct_order_ct_pct = (pd.crosstab(df['genericProduct'], df['order'], normalize='index') * 100).round(2)
genericProduct_EDA_df = pd.DataFrame({
    'total_count': genericProduct_counts,
    'total_percentage': genericProduct_percentages,
    'order_0_count': genericProduct_order_ct[0],
    'order_1_count': genericProduct_order_ct[1],
    'order_0_percentage': genericProduct_order_ct_pct[0],
    'order_1_percentage': genericProduct_order_ct_pct[1]
})
print(genericProduct_EDA_df)
genericProduct_EDA_df.to_csv('../data/processed/genericProduct_EDA.csv', index=True)

# Visualization
plot = sns.countplot(x="genericProduct", hue="order", data=df)
plot.get_yaxis().get_major_formatter().set_scientific(False)
plt.title("Distribution of Generic Product Flag")
plt.savefig('../reports/figures/genericProduct_EDA.png')
plt.show()


# #### ***Sales Index***


# Numbers
salesIndex_counts = df['salesIndex'].value_counts()
salesIndex_percentages = (df['salesIndex'].value_counts(normalize=True) * 100).round(2)
salesIndex_order_ct = pd.crosstab(df['salesIndex'], df['order'])
salesIndex_order_ct_pct = (pd.crosstab(df['salesIndex'], df['order'], normalize='index') * 100).round(2)
salesIndex_EDA_df = pd.DataFrame({
    'total_count': salesIndex_counts,
    'total_percentage': salesIndex_percentages,
    'order_0_count': salesIndex_order_ct[0],
    'order_1_count': salesIndex_order_ct[1],
    'order_0_percentage': salesIndex_order_ct_pct[0],
    'order_1_percentage': salesIndex_order_ct_pct[1]
})
print(salesIndex_EDA_df)
salesIndex_EDA_df.to_csv('../data/processed/salesIndex_EDA.csv', index=True)

# Visualization
plot = sns.countplot(x="salesIndex", hue="order", data=df)
plot.get_yaxis().get_major_formatter().set_scientific(False)
plt.title("Distribution of Sales Index")
plt.savefig('../reports/figures/salesIndex_EDA.png')
plt.show()


# #### ***Campaign Index***


# Numbers
campaignIndex_counts = df['campaignIndex'].value_counts()
campaignIndex_percentages = (df['campaignIndex'].value_counts(normalize=True) * 100).round(2)
campaignIndex_order_ct = pd.crosstab(df['campaignIndex'], df['order'])
campaignIndex_order_ct_pct = (pd.crosstab(df['campaignIndex'], df['order'], normalize='index') * 100).round(2)
campaignIndex_EDA_df = pd.DataFrame({
    'total_count': campaignIndex_counts,
    'total_percentage': campaignIndex_percentages,
    'order_0_count': campaignIndex_order_ct[0],
    'order_1_count': campaignIndex_order_ct[1],
    'order_0_percentage': campaignIndex_order_ct_pct[0],
    'order_1_percentage': campaignIndex_order_ct_pct[1]
})
print(campaignIndex_EDA_df)
campaignIndex_EDA_df.to_csv('../data/processed/campaignIndex_EDA.csv', index=True)

# Visualization
plot = sns.countplot(x="campaignIndex", hue="order", data=df)
plot.get_yaxis().get_major_formatter().set_scientific(False)
plt.title("Distribution of Campaign Index Flag")
plt.savefig('../reports/figures/campaignIndex_EDA.png')
plt.show()


# ### Numerical Features


# #### ***Price***


# Numbers
price_counts = df['price'].value_counts()
price_percentages = (df['price'].value_counts(normalize=True) * 100).round(2)
price_order_ct = pd.crosstab(df['price'], df['order'])
price_order_ct_pct = (pd.crosstab(df['price'], df['order'], normalize='index') * 100).round(2)
price_EDA_df = pd.DataFrame({
    'total_count': price_counts,
    'total_percentage': price_percentages,
    'order_0_count': price_order_ct[0],
    'order_1_count': price_order_ct[1],
    'order_0_percentage': price_order_ct_pct[0],
    'order_1_percentage': price_order_ct_pct[1]
})
print(price_EDA_df)
price_EDA_df.to_csv('../data/processed/price_EDA.csv', index=True)
print(df['price'].describe().apply('{0:.2f}'.format))

sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(data=df, x='price', hue='order', bins=80, multiple='stack', palette='tab10', ax=axes[0])
axes[0].set_title("Distribution of Price")
sns.boxplot(data=df, x='price', y='order', hue='order', palette='tab10', orient='h', ax=axes[1], legend=False)
axes[1].set_title("Boxplot of Price")
plt.tight_layout()
plt.savefig('../reports/figures/price_EDA.png')
plt.show()


# #### ***Revenue***


# Numbers
revenue_counts = df['revenue'].value_counts()
revenue_percentages = (df['revenue'].value_counts(normalize=True) * 100).round(2)
revenue_order_ct = pd.crosstab(df['revenue'], df['order'])
revenue_order_ct_pct = (pd.crosstab(df['revenue'], df['order'], normalize='index') * 100).round(2)
revenue_EDA_df = pd.DataFrame({
    'total_count': revenue_counts,
    'total_percentage': revenue_percentages,
    'order_0_count': revenue_order_ct[0],
    'order_1_count': revenue_order_ct[1],
    'order_0_percentage': revenue_order_ct_pct[0],
    'order_1_percentage': revenue_order_ct_pct[1]
})
print(revenue_EDA_df)
revenue_EDA_df.to_csv('../data/processed/revenue_EDA.csv', index=True)
print(df['revenue'].describe().apply('{0:.2f}'.format))

sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(data=df, x='revenue', hue='order', bins=80, multiple='stack', palette='tab10', ax=axes[0])
axes[0].set_title("Distribution of Revenue")
sns.boxplot(data=df, x='revenue', y='order', hue='order', palette='tab10', orient='h', ax=axes[1], legend=False)
axes[1].set_title("Boxplot of Revenue")
plt.tight_layout()
plt.savefig('../reports/figures/revenue_EDA.png')
plt.show()


# #### ***Reference Price***


# Numbers
rrp_counts = df['rrp'].value_counts()
rrp_percentages = (df['rrp'].value_counts(normalize=True) * 100).round(2)
rrp_order_ct = pd.crosstab(df['rrp'], df['order'])
rrp_order_ct_pct = (pd.crosstab(df['rrp'], df['order'], normalize='index') * 100).round(2)
rrp_EDA_df = pd.DataFrame({
    'total_count': rrp_counts,
    'total_percentage': rrp_percentages,
    'order_0_count': rrp_order_ct[0],
    'order_1_count': rrp_order_ct[1],
    'order_0_percentage': rrp_order_ct_pct[0],
    'order_1_percentage': rrp_order_ct_pct[1]
})
print(rrp_EDA_df)
rrp_EDA_df.to_csv('../data/processed/rrp_EDA.csv', index=True)
print(df['rrp'].describe().apply('{0:.2f}'.format))

sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(data=df, x='rrp', hue='order', bins=80, multiple='stack', palette='tab10', ax=axes[0])
axes[0].set_title("Distribution of RRP")
sns.boxplot(data=df, x='rrp', y='order', hue='order', palette='tab10', orient='h', ax=axes[1], legend=False)
axes[1].set_title("Boxplot of RRP")
plt.tight_layout()
plt.savefig('../reports/figures/rrp_EDA.png')
plt.show()


# #### ***Competitor Price***


# Numbers
competitorPrice_counts = df['competitorPrice'].value_counts()
competitorPrice_percentages = (df['competitorPrice'].value_counts(normalize=True) * 100).round(2)
competitorPrice_order_ct = pd.crosstab(df['competitorPrice'], df['order'])
competitorPrice_order_ct_pct = (pd.crosstab(df['competitorPrice'], df['order'], normalize='index') * 100).round(2)
competitorPrice_EDA_df = pd.DataFrame({
    'total_count': competitorPrice_counts,
    'total_percentage': competitorPrice_percentages,
    'order_0_count': competitorPrice_order_ct[0],
    'order_1_count': competitorPrice_order_ct[1],
    'order_0_percentage': competitorPrice_order_ct_pct[0],
    'order_1_percentage': competitorPrice_order_ct_pct[1]
})
print(competitorPrice_EDA_df)
competitorPrice_EDA_df.to_csv('../data/processed/competitorPrice_EDA.csv', index=True)
print(df['competitorPrice'].describe().apply('{0:.2f}'.format))

sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(data=df, x='competitorPrice', hue='order', bins=80, multiple='stack', palette='tab10', ax=axes[0])
axes[0].set_title("Distribution of Competitor Price")
sns.boxplot(data=df, x='competitorPrice', y='order', hue='order', palette='tab10', orient='h', ax=axes[1], legend=False)
axes[1].set_title("Boxplot of Competitor Price")
plt.tight_layout()
plt.savefig('../reports/figures/competitorPrice_EDA.png')
plt.show()


# # 4 Data Preparation
# 
# Before modelling we must:
# 1. Execute the missing-value handling.
# 2. Engineer new features (like units_bought, price_ratio).
# 3. Perform feature selection and remove leaked features (e.g. basket, click, revenue).
# 4. Perform a stratified train / validation split (80 / 20) on the final selected features.
# 5. Encode high-cardinality categoricals and Scale numericals.
# 
# It is critical that all encoders and scalers are fit **only** on training data to prevent data leakage.


# ## 4.1 Handling Missing Values


# Our EDA has identified missing values in category, campaignIndex, pharmForm, and competitorPrice. We need to decide on a strategy for each:
# - Imputation: Fill missing values with a statistic like the mean, median, or mode.
# - Removal: Remove rows with missing values (use with caution, as it can lead to data loss).
# - Model-based imputation: Use a model to predict the missing values.
# - Create a new category: For categorical variables, you could treat "missing" as a new category.


# Analyse missing values to determine how to handle them in the next steps. sort by missing percentage in descending order
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df) * 100).round(2)
missing_df = pd.DataFrame({'missing_count': missing_values, 'missing_percentage': missing_percentage})
missing_df = missing_df.sort_values(by='missing_percentage', ascending=False)
print(missing_df)


# Analyse if the absence of a value in each column might be correlated with whether an order is placed or not incl. percentage
missing_order_correlation = {}
for column in df.columns:
    if df[column].isnull().sum() > 0:
        missing_order_correlation[column] = df.groupby(df[column].isnull())['order'].mean().round(4) * 100
missing_order_correlation_df = pd.DataFrame(missing_order_correlation).T
missing_order_correlation_df.columns = ['order_percentage_when_missing', 'order_percentage_when_present']
missing_order_correlation_df = missing_order_correlation_df.sort_values(by='order_percentage_when_missing', ascending=False)
print(missing_order_correlation_df)


# ### campainIndex


# With 93.93% of its values missing, the campaignIndex column is mostly empty. This significantly limits its usefulness. Standard imputation methods like using the mean, median, or mode are not suitable here, as filling over 90% of the data with a single value would heavily distort the feature's distribution and likely mislead the model.
# 
# The crosstabulation shows a clear difference in ordering behavior depending on whether campaignIndex is missing.
# - When campaignIndex has a value: The order rate is approximately 27.45% (128,467 orders / 468,035 total).
# - When campaignIndex is missing: The order rate is approximately 25.19% (576,623 orders / 2,287,968 total).
# 
# While there is a slight difference, the fact that the column is overwhelmingly empty suggests that its predictive power is likely low. The information it provides is minimal because for the vast majority of the data, the only signal is "the value is missing."
# 
# Given the extremely high percentage of missing data, the best course of action is to drop the campaignIndex column entirely.
# 
# Reasoning:
# 1. Low Information Content: A feature that is empty 94% of the time is unlikely to provide significant predictive value to a model.
# 2. Risk of Noise: Keeping the column, even by creating a "missing" category, could introduce noise and complexity to the model for very little gain.
# 3. Simplification: Removing the column simplifies the feature set, which can lead to a more robust and interpretable model.


# Drop the 'campaignIndex' column from the DataFrame
df.drop(columns=['campaignIndex'], inplace=True)

# Verify that the column has been dropped by checking the remaining columns
print("Verifying 'campaignIndex' column removal. Remaining columns are:")
print(df.columns)


# ### pharmForm


# The `pharmForm` column has a missing value rate of 7.04%, which is a moderate but significant portion of the data. Simply dropping these rows could discard valuable information and potentially introduce bias if the missingness is not random.
# 
# The provided crosstabulation against the `order` variable is highly revealing:
# -   **Order Rate when `pharmForm` is Missing**: 26.02%
# -   **Order Rate when `pharmForm` is Present**: 19.79%
# 
# This disparity indicates that the missingness in `pharmForm` is **not random**; it is informative. Products with a missing `pharmForm` have a noticeably higher propensity to be ordered. This suggests that the absence of this feature is a predictive signal in itself.
# 
# There could be several underlying business reasons for this pattern:
# 1.  **Product Type**: Certain types of products that sell well might not have a standard pharmaceutical form (e.g., cosmetics, health devices, or bundled items).
# 2.  **Data Entry Practices**: It's possible that for certain popular or fast-moving items, data entry for this specific field is deprioritized, leading to missing values.
# 3.  **Customer Perception**: The absence of a specific pharmaceutical form might make a product appear more like a general consumer good rather than a specialized medical item, which could broaden its appeal and increase sales.
# 
# **Recommended Strategy: Imputation with a 'Missing' Category**
# 
# Given that the missingness is a predictive signal, the best approach is to preserve this information. I recommend imputing the `NaN` values with a distinct string, such as `'Missing'`.
# 
# **Justification:**
# -   **Preserves Information**: This method explicitly captures the signal that the absence of `pharmForm` provides, allowing the machine learning model to learn the relationship between "missingness" and the higher order rate.
# -   **Avoids Data Loss**: It prevents the loss of 7.04% of the rows in your dataset, which would happen if you chose to drop them.
# -   **Prevents Distortion**: Unlike mean/median/mode imputation, creating a new category does not distort the distribution of existing `pharmForm` categories. It treats the missing data as a unique group, which, according to the data, it is.
# 
# This strategy effectively turns the missing data into a useful feature for the model.


# Impute missing values in 'pharmForm' with the string 'Missing'
df.fillna({'pharmForm': 'Missing'}, inplace=True)

# Verify that there are no more missing values in 'pharmForm'
print("Missing values in 'pharmForm' after imputation:", df['pharmForm'].isnull().sum())


# ### category


# Based on the EDA analysis, the `category` column has a high number of unique values and a long-tail distribution. The code also calculates the percentage of missing values for each column.
# As the percentage of missing `category` values is small, we could consider imputing them with the mode (most frequent category). However, given its high cardinality, a better approach is to treat "missing" as a separate category. This can even be a useful predictive signal, as the absence of a category might be correlated with whether an order is placed.
# 
# Why this is the best strategy:
# 1. Preserves Data: 3.17% is a significant amount of data to simply delete. Removing these rows could eliminate valuable information from other columns.
# 2. Avoids Skewing the Distribution: The category feature has high cardinality (many unique values) and a long-tail distribution. Imputing with the most frequent category (the mode) would artificially inflate its importance and might mislead the model.
# 3. Potential Predictive Power: The fact that a category is missing might itself be a useful signal for the model. For example, products without a category might have a different ordering behavior. Creating a separate "Missing" or "Unknown" category allows the model to learn this potential relationship.


# Impute missing values in 'category' with 'Missing'
df.fillna({'category': 'Missing'}, inplace=True)

# Verify that there are no more missing values in 'category'
print("Missing values in 'category' after imputation:", df['category'].isnull().sum())


# ### competitorPrice


# The `competitorPrice` column has a relatively low missing value rate of 3.65%. While this percentage is small, the relationship between its missingness and the `order` rate is too significant to ignore.
# 
# The crosstabulation reveals a stark contrast:
# -   **Order Rate when `competitorPrice` is Missing**: 25.94%
# -   **Order Rate when `competitorPrice` is Present**: 16.17%
# 
# This is a substantial difference, indicating that products without a listed competitor price are ordered much more frequently. The missingness is clearly **informative** and not random. Dropping these rows would mean discarding a strong predictive signal.
# 
# Several business scenarios could explain this phenomenon:
# 1.  **Exclusive or Unique Products**: The most likely reason is that these products are exclusive to our store. If there are no direct competitors, there is no competitor price to list. Such exclusive items are often highly sought after, leading to a higher purchase rate.
# 2.  **Loss-Leader or "Too Low to Show" Pricing**: The product's price might be so competitive that the business has chosen not to display the competitor's price to avoid direct price wars or to comply with pricing agreements. This aggressive pricing would naturally drive more orders.
# 3.  **New Product Launches**: For newly introduced items, the data collection process for competitor pricing may not have been completed yet. These new products might have high initial sales due to launch promotions or novelty effects.
# 
# We considered using a product's own price or rrp to fill the missing competitorPrice values. While this seems intuitive, we decided against it for a crucial reason: it would fundamentally alter the meaning of the feature.
# The competitorPrice column represents external market data, whereas our price and rrp are internal strategic values. By using our own price for imputation, we would be erasing the powerful signal that the data is missing, which often indicates an exclusive product with no direct competitor. This could mislead the model by creating an artificial relationship in the data.
# Therefore, we secondly considered creating a separate binary flag to capture the "missingness" and then using a neutral value like the median for imputation is a more robust approach that preserves the feature's integrity.
# 
# Instead, because `competitorPrice` is a numerical feature, simply filling it with a value like -1 or 0 could be misinterpreted by some models as a very low price. A more robust approach is to explicitly capture the "missingness" as a feature and then impute the missing price values in a way that doesn't distort the original distribution.
# 
# We choose a two-step strategy:
# 1.  **Create a Binary Flag**: Create a new column, for example `competitorPrice_is_missing`, that is `1` if the `competitorPrice` was missing and `0` otherwise. This directly feeds the powerful "missingness" signal into the model.
# 2.  **Impute the Original Column**: After creating the flag, you can impute the missing values in the original `competitorPrice` column. A good choice would be the **median** competitor price. Using the median is generally more robust to outliers than the mean. This step ensures the column remains a valid numerical feature without `NaN` values.
# 
# **Justification:**
# -   **Captures the Signal**: The binary flag ensures the model can directly learn the strong relationship between the absence of a competitor price and a higher order rate.
# -   **Preserves the Feature**: Imputing the original column allows you to still use `competitorPrice` as a numerical feature (e.g., to calculate price differences), without having to deal with `NaN`s.
# -   **Robustness**: This two-part strategy is more robust and informative than simply filling the missing values with a placeholder like 0 or -1, which could be misinterpreted by the model.
# 
# ------
# 
# After presenting the idea resp. implementation, we revert this approach and simply handle this feature as the fact that "when there is no competitor price, our company has no competitor for the certain product.". We implement the following strategy:
# - Keep the binary which flags, if the value is missing
# - Create a new feature 'competitor_difference' or 'price_ratio' (which we already did)
#     - We proceed this in the chapter 'Feature Creation' (Or here?)


# Create a binary flag for missing 'competitorPrice'
df['competitorPrice_is_missing'] = df['competitorPrice'].isnull().astype(int)

# Calculate the median of 'competitorPrice'
median_competitor_price = df['competitorPrice'].median()

# Impute missing values with the median
df.fillna({'competitorPrice': median_competitor_price}, inplace=True)

# Verify imputation
print("Missing values in 'competitorPrice' after imputation:", df['competitorPrice'].isnull().sum())


# ## 4.2 Feature Engineering


# ### Feature Selection


# #### Line ID
# 
# Because the line ID is not providing any addition information, we drop the column.


# Drop the 'lineId' column from the DataFrame
df.drop(columns=['lineID'], inplace=True)

# Verify that the column has been dropped by checking the remaining columns
print("Verifying 'lineId' column removal. Remaining columns are:")
print(df.columns)


# ### 4.2.1 Feature Creation


# ### Unit(s) Bought


# To address the project's objective of predicting not only whether a product is purchased but also in what quantity, we needed to engineer a target variable for the quantity. The raw data provides revenue and price for each transaction but does not explicitly state the number of units sold.
# 
# We derived the units_bought feature by dividing the revenue by the price. This calculation gives us the exact number of units purchased for each order event. For instances where no order occurred (order == 0), the revenue is zero, which correctly results in zero units bought. As this new feature is created by combining existing features, it is a synthetic feature.
# 
# To ensure data integrity and prepare the feature for modeling, we handled potential floating-point inaccuracies by rounding the result to the nearest whole number and converting the data type to an integer. This new units_bought column will serve as the target for our regression model, which aims to predict the quantity of a purchase.


# We fill any potential NaN values (e.g., from 0/0) with 0 and handle division by zero
df['units_bought'] = (df['revenue'] / df['price']).round().fillna(0)

# Convert the column to an integer data type for cleanliness
df['units_bought'] = df['units_bought'].astype(int)

# Display the first few rows with the new column and its data type
print(df[['price', 'revenue', 'order', 'units_bought']].head())
print("\nData type of 'units_bought':", df['units_bought'].dtype)

# Verify the distribution of units bought for actual orders
print("\nValue counts of units_bought for orders > 0:")
print(df[df['order'] == 1]['units_bought'].value_counts().head())
print(df.head())

df['units_bought'].describe()
df['units_bought'].hist()


# ### Price Ratio


# We fill any potential NaN values with 1.0 (price parity) and handle division by zero
df['price_ratio'] = (
    df['price'] / df['competitorPrice'].replace(0, np.nan)  # 0 → NaN before division
).fillna(1.0)                                               # NaN (from 0-denom) → 1.0

# Convert the column to a float data type for cleanliness
df['price_ratio'] = df['price_ratio'].astype(float)

# Display the first few rows with the new column and its data type
print(df[['price', 'competitorPrice', 'revenue', 'order', 'price_ratio']].head())
print("\nData type of 'price_ratio':", df['price_ratio'].dtype)

# Verify the feature for actual orders
print("\nDescriptive statistics of price_ratio for orders > 0:")
print(df[df['order'] == 1]['price_ratio'].describe())
print(df.head())


# ### Interaction Type


print(df.columns)


# We merge the features click, basket, order into one new feature 'interaction_type' to capture the type of interaction in a single column
def determine_interaction_type(row):
    if row['order'] == 1:
        return 'order'
    elif row['basket'] == 1:
        return 'basket'
    elif row['click'] == 1:
        return 'click'
    else:
        return 'none'

df['interaction_type'] = df.apply(determine_interaction_type, axis=1)

# Display the first few rows with the new column and its data type
print(df[['interaction_type']].head())
print("\nData type of 'interaction_type':", df['interaction_type'].dtype)

# Drop the original 'click' and 'basket' columns as they are now represented in 'interaction_type'.
# We keep 'order' as it is the target variable
df.drop(columns=['click', 'basket'], inplace=True)

# Verify that the columns have been dropped and the new column is present
print("Verifying column changes. Remaining columns are:")
print(df.columns)


# ### 4.2.2 Feature Transformation


# =============================================================================
# UNIFIED DATA PREPARATION PIPELINE
# =============================================================================
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
import pandas as pd
import numpy as np

# Balancing 'order' classes (Undersampling majority class - non-purchases)
df_majority = df[df['order'] == 0]
df_minority = df[df['order'] == 1]

df_majority_downsampled = resample(df_majority, 
                                 replace=False,    
                                 n_samples=len(df_minority), 
                                 random_state=42) 
df_balanced = pd.concat([df_majority_downsampled, df_minority])

# Sort by day to keep chronological order valid for TimeSeriesSplit
if 'day' in df_balanced.columns:
    df_balanced.sort_values('day', inplace=True)

# Let's define the targets. First task: predict 'order'.
X = df_balanced.drop(columns=['order'])
y = df_balanced['order']

# We won't use train_test_split here, we use TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# For the explicit train/test split, we can just take the last split or a chronological split.
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

# Encoding Categorical Variables
categorical_cols = ['pid', 'manufacturer', 'content', 'group', 'pharmForm', 'category', 'adFlag', 'availability', 'genericProduct', 'unit', 'interaction_type']
categorical_cols = [c for c in categorical_cols if c in X_train.columns]

# We will use Label Encoding for simplicity, but we need to handle unseen categories in the test set.
# We will create a mapping dictionary for each categorical column to ensure that unseen categories in the test set
encoders = {}
for col in categorical_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])

# Create a mapping dictionary for the test set to handle unseen categories
    d = dict(zip(le.classes_, le.transform(le.classes_)))
    X_test[col] = X_test[col].apply(lambda x: d.get(x, -1)) 
    encoders[col] = le

# Scaling Numerical Variables
numerical_cols = [c for c in X_train.columns if c not in categorical_cols]

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols]  = scaler.transform(X_test[numerical_cols])

feature_names = X_train.columns
X_train_scaled = X_train.values
X_test_scaled = X_test.values

print("Training set shape:", X_train_scaled.shape)
print("Testing set shape:", X_test_scaled.shape)


# # 5. Modeling


# --- 5. Model training and feature importance with TimeSeriesSplit ---
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def get_importance_with_tscv(model, X, y, model_name, cv_splits=5):
    # Use TimeSeriesSplit instead of standard CV
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    # Get cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='f1')

    # Get feature importance
    model.fit(X, y)
    if model_name in ['Decision Tree', 'Random Forest', 'XGBoost']:
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])

    return pd.Series(importances, index=feature_names, name=model_name), cv_scores.mean()

models_classification = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# Note: since we balanced the classes with undersampling, we do not need class_weight='balanced'
top_features_by_model = {}
cv_scores = {}
all_importances = []

for name, model in models_classification.items():
    print(f"Training Classification Model: {name}...")
    imp, cv_score = get_importance_with_tscv(model, X_train_scaled, y_train, name)
    top_features = imp.sort_values(ascending=False).head(10)
    top_features_by_model[name] = top_features
    cv_scores[name] = cv_score
    all_importances.append(imp)

print("\nTimeSeriesSplit CV F1-Scores for 'order':")
for name, score in cv_scores.items():
    print(f"{name}: {score:.4f}")

print("\nTop 10 Features by Model:")
for model_name, features in top_features_by_model.items():
    print(f"\n{model_name}:")
    print(features)


# --- Regression for Revenue ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("\n--- Regression for Revenue ---")

# Let's predict revenue for the cases where an order actually occurred (or we can predict for everything)
# Typically, predicting revenue makes sense for positive orders.
df_revenue = df[df['order'] == 1].copy()

# Sort by day
if 'day' in df_revenue.columns:
    df_revenue.sort_values('day', inplace=True)

# Target
y_reg = df_revenue['revenue']
# Drop target-related metrics to prevent leakage in regression
cols_to_drop_reg = ['revenue', 'units_bought'] 
# We keep order, basket, click, funnel_status inside as requested, order is just 1.
X_reg = df_revenue.drop(columns=[col for col in cols_to_drop_reg if col in df_revenue.columns])

split_idx_reg = int(len(X_reg) * 0.8)
X_train_reg, X_test_reg = X_reg.iloc[:split_idx_reg].copy(), X_reg.iloc[split_idx_reg:].copy()
y_train_reg, y_test_reg = y_reg.iloc[:split_idx_reg].copy(), y_reg.iloc[split_idx_reg:].copy()

# 4. Encoding Categorical Variables
categorical_cols_reg = [c for c in categorical_cols if c in X_train_reg.columns]

encoders_reg = {}
for col in categorical_cols_reg:
    X_train_reg[col] = X_train_reg[col].astype(str)
    X_test_reg[col] = X_test_reg[col].astype(str)
    le = LabelEncoder()
    X_train_reg[col] = le.fit_transform(X_train_reg[col])
    d = dict(zip(le.classes_, le.transform(le.classes_)))
    X_test_reg[col] = X_test_reg[col].apply(lambda x: d.get(x, -1)) 
    encoders_reg[col] = le

# 5. Scaling Numerical Variables
numerical_cols_reg = [c for c in X_train_reg.columns if c not in categorical_cols_reg]

scaler_reg = StandardScaler()
X_train_scaled_reg = X_train_reg.copy()
X_test_scaled_reg = X_test_reg.copy()

X_train_scaled_reg[numerical_cols_reg] = scaler_reg.fit_transform(X_train_reg[numerical_cols_reg])
X_test_scaled_reg[numerical_cols_reg]  = scaler_reg.transform(X_test_reg[numerical_cols_reg])

models_regression = {
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost Regressor': xgb.XGBRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression()
}

print("Training Regression Models for 'revenue'...")
for name, model in models_regression.items():
    model.fit(X_train_scaled_reg, y_train_reg)
    preds = model.predict(X_test_scaled_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, preds))
    mae = mean_absolute_error(y_test_reg, preds)
    print(f"\n{name} Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")



