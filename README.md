# ❤️ Heart Disease Prediction Project

A smart web app that predicts heart disease risk using machine learning. Just enter your health info and get instant results with beautiful visualizations!

## 🚀 Quick Start (Super Easy!)

**Step 1:** Clone this project
```bash
git clone https://github.com/Seifeldin-Amr/Heart-Disease-Project.git
cd Heart-Disease-Project
```

**Step 2:** Install everything you need
```bash
pip install -r requirements.txt
```

**Step 3:** Run the app (creates public link automatically!)
```bash
cd ui
python app.py
```

That's it! The app will automatically:
- Start the web interface
- Create a public URL you can share with anyone
- Show you both local and public links in the terminal

## 🌐 Public Access

When you run `python app.py`, you'll see something like this in your terminal:

```
🚀 Auto-launching Heart Disease Prediction App...
📱 Creating public URL automatically...
✅ SUCCESS! App is now live:
🌐 PUBLIC URL: https://amazing-app-name.ngrok-free.dev
📍 LOCAL URL:  http://localhost:8501
```

Share the **PUBLIC URL** with anyone - no installation needed on their end!

## 🎯 What This App Does

### For Users:
- **Easy Input**: Fill out a simple form with your health information
- **Instant Results**: Get your heart disease risk prediction in seconds  
- **Smart Analysis**: Uses 5 levels of severity (No Disease → Very Severe)
- **Beautiful Charts**: Explore health data with interactive visualizations
- **Confidence Score**: See how confident the AI is about your result
- **Dataset Explorer**: Browse the heart disease dataset with interactive charts
- **Feature Analysis**: See which health factors matter most
- **Correlation Maps**: Understand relationships between different health metrics

## 📊 Project Structure

```
Heart-Disease-Project/
├── ui/
│   ├── app.py              # 🎯 Main app (run this!)
│   └── requirements.txt    # 📦 What you need to install
├── data/
│   └── heart_disease.csv   # 💾 Heart disease dataset
├── models/
│   └── final_model.pkl     # 🤖 Trained AI model
├── notebooks/              # 📚 Data science work
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb  
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
├── results/
│   └── evaluation_metrics.txt
└── deployment/
    └── ngrok_setup.txt     # 🌐 Deployment guide
```

### The 10 Key Health Factors:
- Cholesterol level
- Age  
- Maximum heart rate
- Number of major vessels (0-3)
- Exercise-induced chest pain
- Chest pain type
- Heart rhythm during exercise
- Blood flow test results


## Requirements
- Python 3.8+
- All packages in `requirements.txt`
- Ngrok account (free) for public URLs

## Manual Setup
If the auto-setup doesn't work:

```bash
# Install ngrok
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok

# Add your ngrok token (get from https://dashboard.ngrok.com)
ngrok config add-authtoken YOUR_TOKEN_HERE

# Run manually
streamlit run ui/app.py
```

## 📈 Model Performance

The AI model achieves:
- **60.7%** accuracy on test data
- **78.3%** ROC-AUC score (good at ranking risk levels)
- Works with **10 carefully selected features** out of 13 original ones


