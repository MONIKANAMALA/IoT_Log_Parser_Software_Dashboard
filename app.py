from flask import Flask, render_template
import io
import base64
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Use a non-interactive backend
matplotlib.use('Agg')

app = Flask(__name__)

# Safe regex search function
def safe_search(pattern, text):
    match = re.search(pattern, text)
    return match.group(1) if match else None

# Parse log file and return cleaned DataFrame
def parse_log_file():
    log_file_path = 'assignment_prod.log' 
    with open(log_file_path, 'r') as file:
        log_content = file.read()

    # Extract plain logs
    plain_data_pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}) (.*)'
    plain_data = re.findall(plain_data_pattern, log_content)

    df = pd.DataFrame(plain_data, columns=["timestamp", "log_message"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=["timestamp"])  # Remove invalid timestamps

    # Add derived fields
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.day_name()
    df['user'] = df['log_message'].apply(lambda x: x.split()[0] if len(x.split()) > 1 else 'unknown')
    df['log_message_length'] = df['log_message'].apply(len)
    df['action_type'] = df['log_message'].apply(lambda x: safe_search(r'action=([^ ]*)', x) or 'login')
    return df

def create_plot(plot_func, *args, **kwargs):
    figsize = kwargs.pop("figsize", (8, 5))
    title = kwargs.pop("title", None)
    xlabel = kwargs.pop("xlabel", None)
    ylabel = kwargs.pop("ylabel", None)

    plt.figure(figsize=figsize)
    plot_func(*args, **kwargs)

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches="tight")
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode('utf8')

@app.route('/')
def dashboard():
    df = parse_log_file()
    plots = []

    # 1. Log Count Distribution (Box Plot)
    plots.append(create_plot(sns.boxplot, x=df['hour'], y=df['log_message_length'], 
                             title="Log Count Distribution by Hour", xlabel="Hour", ylabel="Log Message Length"))

    # 2. Bar Plot of Average Log Message Length by Weekday
    avg_length_by_weekday = df.groupby('weekday')['log_message_length'].mean().reindex([
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ])
    plots.append(create_plot(avg_length_by_weekday.plot.bar, color='skyblue', title="Average Log Message Length by Weekday",
                             xlabel="Weekday", ylabel="Average Log Message Length"))

    # 3. Grouped Bar Plot for Action Types by Weekday
    weekday_action_counts = df.groupby(['weekday', 'action_type']).size().unstack(fill_value=0)
    weekday_action_counts = weekday_action_counts.reindex([
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ])
    plots.append(create_plot(weekday_action_counts.plot.bar, figsize=(10, 6), title="Action Types by Weekday",
                             xlabel="Weekday", ylabel="Count"))

    # 4. Correlation Matrix (Heatmap)
    correlation_data = df[['hour', 'log_message_length']].corr()
    plots.append(create_plot(sns.heatmap, correlation_data, annot=True, cmap='viridis', 
                             title="Correlation Matrix"))

    # 5. Line Plot for Daily Log Count Trend
    daily_logs = df.set_index('timestamp').resample('D').size()
    plots.append(create_plot(daily_logs.plot, title="Daily Log Count Trend", xlabel="Date", ylabel="Log Count"))

    # 6. KDE Plot for Log Message Length
    plots.append(create_plot(sns.kdeplot, df['log_message_length'], shade=True, color="purple", 
                             title="Density of Log Message Lengths", xlabel="Log Message Length", ylabel="Density"))

    # 7. Parallel Coordinate Plot for Hour and Log Message Length
    parallel_data = df[['hour', 'log_message_length']].copy()
    parallel_data['hour_category'] = pd.cut(parallel_data['hour'], bins=4, labels=["Early", "Morning", "Afternoon", "Night"])
    sns.set_palette("husl")
    plots.append(create_plot(sns.violinplot, x='hour_category', y='log_message_length', data=parallel_data, 
                             title="Message Length by Time Period"))

    return render_template('dataplots.html', plot_images=plots)

if __name__ == '__main__':
    app.run(debug=True)
