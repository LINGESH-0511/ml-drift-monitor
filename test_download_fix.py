import streamlit as st
import pandas as pd
import io
from datetime import datetime

st.set_page_config(page_title="Download Test", layout="wide")

st.title("Download Test - CredGuard")

# Create sample data
df = pd.DataFrame({
    'Feature': ['income', 'credit_score', 'age', 'employment_type'],
    'PSI': [0.324, 0.287, 0.042, 0.156],
    'Severity': ['HIGH', 'HIGH', 'LOW', 'MEDIUM']
})

# Method 1: Direct download without button click
st.write("### Test 1: Direct Download")
csv_data = df.to_csv(index=False)
st.download_button(
    label="Download CSV Directly",
    data=csv_data,
    file_name=f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

# Method 2: JSON download
st.write("### Test 2: JSON Download")
json_data = df.to_json(orient='records', indent=2)
st.download_button(
    label="Download JSON Directly",
    data=json_data,
    file_name=f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    mime="application/json"
)

st.write("### Instructions:")
st.info("""
1. Click the download buttons above
2. Files should download immediately without page refresh
3. If downloads work, the main app will work too
""")
