import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="ProcureAI | Enterprise Procurement & Inventory Engine",
    page_icon="📦",
    layout="wide"
)

st.title("📦 ProcureAI — Enterprise Inventory & Procurement Intelligence")
st.caption("Upload any inventory, order, or supply chain dataset to calculate Safety Stock, Reorder Points (ROP), EOQ, ABC Classification, and Automated Purchase Orders.")
st.markdown("---")

st.sidebar.header("📁 Data Input & Mapping Hub")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel dataset", type=["csv", "xlsx", "xls"])

DEFAULT_DATA = Path("data/selected_dataset.csv")

@st.cache_data
def load_data(file):
    if file is not None:
        if file.name.endswith(".csv"):
            raw_df = pd.read_csv(file)
        else:
            raw_df = pd.read_excel(file)
    elif DEFAULT_DATA.exists():
        raw_df = pd.read_csv(DEFAULT_DATA)
    else:
        return None
    
    cleaned_columns = []
    for i, col in enumerate(raw_df.columns):
        col_str = str(col).strip()
        if col_str == "" or col_str.startswith("Unnamed"):
            cleaned_columns.append(f"Column_{i+1}")
        else:
            cleaned_columns.append(col_str)
    raw_df.columns = cleaned_columns
    return raw_df

df = load_data(uploaded_file)

if df is None:
    st.warning("⚠️ No dataset loaded. Please upload a CSV/Excel file via the sidebar, or ensure the default dataset exists.")
    st.stop()

def find_best_column(columns, keywords):
    for col in columns:
        col_lower = str(col).lower()
        for kw in keywords:
            if kw in col_lower:
                return col
    return columns[0] if len(columns) > 0 else None

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Smart Column Mapping")
all_cols = df.columns.tolist()

default_item = find_best_column(all_cols, ["item", "sku", "product", "name", "material", "order", "system"])
default_stock = find_best_column(all_cols, ["stock", "inventory", "qty", "quantity", "on_hand"])
default_usage = find_best_column(all_cols, ["usage", "demand", "daily", "sales", "velocity"])
default_lead = find_best_column(all_cols, ["lead", "delivery", "shipping", "days", "score"])
default_cost = find_best_column(all_cols, ["cost", "price", "value", "unit", "amount"])

item_col = st.sidebar.selectbox("Item / SKU Column", all_cols, index=all_cols.index(default_item) if default_item in all_cols else 0)
stock_col = st.sidebar.selectbox("Current Stock Column", all_cols, index=all_cols.index(default_stock) if default_stock in all_cols else min(1, len(all_cols)-1))
usage_col = st.sidebar.selectbox("Daily Demand Column", all_cols, index=all_cols.index(default_usage) if default_usage in all_cols else min(2, len(all_cols)-1))
lead_col = st.sidebar.selectbox("Supplier Lead Time Column", all_cols, index=all_cols.index(default_lead) if default_lead in all_cols else min(3, len(all_cols)-1))
cost_col = st.sidebar.selectbox("Unit Cost Column", all_cols, index=all_cols.index(default_cost) if default_cost in all_cols else min(4, len(all_cols)-1))

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Inventory Parameters")
ordering_cost = st.sidebar.number_input("Standard Ordering Cost per PO ($)", min_value=10.0, max_value=500.0, value=50.0)
holding_cost_pct = st.sidebar.slider("Annual Holding Cost (% of Item Cost)", min_value=5.0, max_value=40.0, value=20.0) / 100.0

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Overview", 
    "🔍 ABC Classification & Inspector", 
    "⚡ Automated ROP & EOQ Plan", 
    "📋 Purchase Order Generator",
    "📈 Analytics & Visuals"
])

with tab1:
    st.subheader("High-Level Procurement KPIs")
    
    col1, col2, col3, col4 = st.columns(4)
    total_items = len(df)
    
    try:
        s_vals = pd.to_numeric(df[stock_col], errors='coerce').fillna(0)
        c_vals = pd.to_numeric(df[cost_col], errors='coerce').fillna(0)
        total_inventory_value = (s_vals * c_vals).sum()
    except:
        total_inventory_value = 0.0
        
    try:
        avg_lead_time = pd.to_numeric(df[lead_col], errors='coerce').mean()
    except:
        avg_lead_time = 0.0
    
    col1.metric("Total Items / SKUs", f"{total_items:,}")
    col2.metric("Total Inventory Asset Value", f"${total_inventory_value:,.2f}")
    col3.metric("Average Supplier Lead Time", f"{avg_lead_time:.1f} Days")
    col4.metric("Active Data Columns", len(df.columns))

    st.markdown("---")
    st.write("### Quick Inventory Preview")
    st.dataframe(df.head(5), use_container_width=True)

with tab2:
    st.subheader("🔍 ABC Inventory Classification (Pareto Analysis)")
    st.write("ABC classification categorizes inventory into **Class A** (high value, strict control), **Class B** (moderate value), and **Class C** (bulk low value).")

    try:
        analysis_df = df.copy()
        u_vals = pd.to_numeric(analysis_df[usage_col], errors='coerce').fillna(0)
        c_vals = pd.to_numeric(analysis_df[cost_col], errors='coerce').fillna(0)
        
        analysis_df["Annual_Demand"] = u_vals * 365
        analysis_df["Total_Annual_Value"] = analysis_df["Annual_Demand"] * c_vals
        
        analysis_df = analysis_df.sort_values(by="Total_Annual_Value", ascending=False).reset_index(drop=True)
        analysis_df["Cumulative_Value"] = analysis_df["Total_Annual_Value"].cumsum()
        total_val = analysis_df["Total_Annual_Value"].sum()
        analysis_df["Cumulative_Pct"] = analysis_df["Cumulative_Value"] / total_val if total_val > 0 else 0
        
        def assign_abc(pct):
            if pct <= 0.80:
                return "Class A (High Value)"
            elif pct <= 0.95:
                return "Class B (Moderate Value)"
            else:
                return "Class C (Bulk Low Value)"
                
        analysis_df["ABC_Class"] = analysis_df["Cumulative_Pct"].apply(assign_abc)
        
        c1, c2, c3 = st.columns(3)
        class_a_count = (analysis_df["ABC_Class"] == "Class A (High Value)").sum()
        class_b_count = (analysis_df["ABC_Class"] == "Class B (Moderate Value)").sum()
        class_c_count = (analysis_df["ABC_Class"] == "Class C (Bulk Low Value)").sum()
        
        c1.metric("Class A Items (Top 80% Value)", f"{class_a_count} SKUs")
        c2.metric("Class B Items (Next 15% Value)", f"{class_b_count} SKUs")
        c3.metric("Class C Items (Remaining 5%)", f"{class_c_count} SKUs")
        
        st.markdown("---")
        search_query = st.text_input("🔍 Search classified inventory:", "")
        if search_query:
            mask = analysis_df.astype(str).apply(lambda x: x.str.contains(search_query, case=False)).any(axis=1)
            filtered_abc = analysis_df[mask]
        else:
            filtered_abc = analysis_df
            
        st.dataframe(filtered_abc, use_container_width=True)
        st.download_button("📥 Download ABC Classified Data as CSV", filtered_abc.to_csv(index=False).encode("utf-8"), "procureai_abc_classified.csv", "text/csv")
        
    except Exception as e:
        st.warning(f"Could not compute ABC classification. Please verify selected numeric columns in the sidebar. Details: {e}")
        st.dataframe(df, use_container_width=True)

with tab3:
    st.subheader("⚡ Automated Reorder Point (ROP) & Economic Order Quantity (EOQ)")
    st.write("Combining **Reorder Point** (when to buy) with **EOQ** (how much to buy to minimize holding & ordering costs).")

    try:
        safety_buffer = st.slider("Safety Stock Buffer (Days)", 1, 14, 3)
        
        calc_df = df.copy()
        u_vals = pd.to_numeric(calc_df[usage_col], errors='coerce').fillna(0)
        l_vals = pd.to_numeric(calc_df[lead_col], errors='coerce').fillna(0)
        s_vals = pd.to_numeric(calc_df[stock_col], errors='coerce').fillna(0)
        c_vals = pd.to_numeric(calc_df[cost_col], errors='coerce').fillna(0)
        
        calc_df["Safety_Stock"] = u_vals * safety_buffer
        calc_df["Reorder_Point"] = (u_vals * l_vals) + calc_df["Safety_Stock"]
        calc_df["Annual_Demand"] = u_vals * 365
        
        holding_cost_per_unit = c_vals * holding_cost_pct
        calc_df["EOQ"] = np.sqrt((2 * calc_df["Annual_Demand"] * ordering_cost) / np.maximum(holding_cost_per_unit, 0.01))
        calc_df["EOQ"] = np.ceil(calc_df["EOQ"]).astype(int)
        
        calc_df["Needs_Reorder"] = s_vals <= calc_df["Reorder_Point"]
        
        reorder_needed = calc_df[calc_df["Needs_Reorder"]]
        
        if len(reorder_needed) > 0:
            st.error(f"🚨 **{len(reorder_needed)} SKUs have breached their Reorder Point and require replenishment!**")
            display_cols = [item_col, stock_col, "Reorder_Point", "Safety_Stock", "EOQ", cost_col]
            st.dataframe(reorder_needed[display_cols], use_container_width=True)
            
            total_po_cost = (reorder_needed["EOQ"] * c_vals.loc[reorder_needed.index]).sum()
            st.info(f"💰 **Estimated Capital Required for Replenishment Orders:** ${total_po_cost:,.2f}")
        else:
            st.success("✅ All inventory levels are healthy above their Reorder Points.")
            
        with st.expander("View Full Calculated ROP & EOQ Matrix"):
            st.dataframe(calc_df, use_container_width=True)
            
    except Exception as e:
        st.warning(f"Error calculating ROP & EOQ. Please ensure your sidebar column selections point to numeric columns. Details: {e}")

with tab4:
    st.subheader("📋 Automated Purchase Order (PO) Generator")
    st.write("Generate and download formal purchase orders for all items that require immediate replenishment.")

    try:
        calc_df = df.copy()
        u_vals = pd.to_numeric(calc_df[usage_col], errors='coerce').fillna(0)
        l_vals = pd.to_numeric(calc_df[lead_col], errors='coerce').fillna(0)
        s_vals = pd.to_numeric(calc_df[stock_col], errors='coerce').fillna(0)
        c_vals = pd.to_numeric(calc_df[cost_col], errors='coerce').fillna(0)
        
        calc_df["Safety_Stock"] = u_vals * 3
        calc_df["Reorder_Point"] = (u_vals * l_vals) + calc_df["Safety_Stock"]
        calc_df["Annual_Demand"] = u_vals * 365
        holding_cost_per_unit = c_vals * holding_cost_pct
        calc_df["EOQ"] = np.ceil(np.sqrt((2 * calc_df["Annual_Demand"] * ordering_cost) / np.maximum(holding_cost_per_unit, 0.01))).astype(int)
        calc_df["Needs_Reorder"] = s_vals <= calc_df["Reorder_Point"]
        
        po_df = calc_df[calc_df["Needs_Reorder"]].copy()
        
        if len(po_df) > 0:
            po_df["PO_Quantity"] = po_df["EOQ"]
            po_df["Total_PO_Cost"] = po_df["PO_Quantity"] * c_vals.loc[po_df.index]
            
            summary_po = po_df[[item_col, stock_col, "Reorder_Point", "PO_Quantity", cost_col, "Total_PO_Cost"]].rename(columns={
                item_col: "Item_SKU",
                stock_col: "Current_Stock",
                cost_col: "Unit_Cost"
            })
            
            st.success(f"Generated {len(summary_po)} purchase orders ready for dispatch.")
            st.dataframe(summary_po, use_container_width=True)
            
            st.download_button(
                "📥 Download Official Purchase Order (CSV)", 
                summary_po.to_csv(index=False).encode("utf-8"), 
                "ProcureAI_Purchase_Orders.csv", 
                "text/csv"
            )
        else:
            st.info("No purchase orders required at this time. All inventory stock is sufficient.")
    except Exception as e:
        st.warning(f"Could not generate PO list: {e}")

with tab5:
    st.subheader("Dynamic Data Visualizations")
    
    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns 
        if not str(col).startswith("Column_")
    ]
    
    if len(numeric_cols) >= 2:
        default_x_idx = 0
        default_y_idx = min(1, len(numeric_cols)-1)
        
        for idx, col in enumerate(numeric_cols):
            col_lower = col.lower()
            if any(k in col_lower for k in ["cost", "price", "lead", "days"]):
                default_x_idx = idx
            if any(k in col_lower for k in ["sales", "demand", "usage", "inventory", "stock"]):
                default_y_idx = idx

        col_x = st.selectbox("Select X-axis metric:", numeric_cols, index=default_x_idx, key="viz_x")
        col_y = st.selectbox("Select Y-axis metric:", numeric_cols, index=default_y_idx, key="viz_y")
        
        chart_df = df.copy()
        chart_df.columns = [str(c).replace(":", r"\:") for c in chart_df.columns]
        safe_col_x = col_x.replace(":", r"\:")
        safe_col_y = col_y.replace(":", r"\:")
        
        st.scatter_chart(chart_df, x=safe_col_x, y=safe_col_y)
    elif len(numeric_cols) == 1:
        col_name = numeric_cols[0]
        val_counts = df[col_name].value_counts().head(20).reset_index()
        val_counts.columns = ["Category", "Count"]
        val_counts["Category"] = val_counts["Category"].astype(str).str.replace(":", r"\:")
        st.bar_chart(val_counts, x="Category", y="Count")
    else:
        st.write("No numeric columns available for charting.")