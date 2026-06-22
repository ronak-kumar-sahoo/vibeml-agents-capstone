import os
import sqlite3
import pandas as pd
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP("VibeML-Data-Server")

@mcp.tool()
def get_dataset_schema(dataset_path: str) -> str:
    """
    Analyzes the structure of a CSV dataset.
    Returns the columns, data types, shape, and count of missing values.
    """
    if not os.path.exists(dataset_path):
        return f"Error: File not found at {dataset_path}"
    
    try:
        df = pd.read_csv(dataset_path)
        rows, cols = df.shape
        dtypes = df.dtypes.to_dict()
        missing = df.isnull().sum().to_dict()
        
        schema_info = [
            f"Dataset Path: {dataset_path}",
            f"Dimensions: {rows} rows, {cols} columns\n",
            "Columns & Types:"
        ]
        for col, dtype in dtypes.items():
            schema_info.append(f" - {col}: {dtype} (Missing values: {missing[col]})")
            
        return "\n".join(schema_info)
    except Exception as e:
        return f"Error reading schema: {str(e)}"


@mcp.tool()
def get_data_summary(dataset_path: str) -> str:
    """
    Generates descriptive statistics for the numerical and categorical columns
    in the dataset.
    """
    if not os.path.exists(dataset_path):
        return f"Error: File not found at {dataset_path}"
    
    try:
        df = pd.read_csv(dataset_path)
        summary = []
        
        # Numerical summary
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if num_cols:
            summary.append("### Numerical Statistics")
            summary.append(df[num_cols].describe().to_markdown())
            summary.append("")
            
        # Categorical summary
        cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        if cat_cols:
            summary.append("### Categorical Statistics")
            cat_summary = []
            for col in cat_cols:
                unique = df[col].nunique()
                top = df[col].mode().tolist()
                top_str = ", ".join(map(str, top[:3]))
                cat_summary.append({
                    "Column": col,
                    "Unique Values": unique,
                    "Most Common": top_str
                })
            summary.append(pd.DataFrame(cat_summary).to_markdown(index=False))
            
        return "\n".join(summary)
    except Exception as e:
        return f"Error calculating statistics: {str(e)}"


@mcp.tool()
def query_data_sql(dataset_path: str, sql_query: str) -> str:
    """
    Runs a SQL query against the CSV dataset.
    The table is named 'data'.
    Example query: 'SELECT churn, COUNT(*), AVG(tenure) FROM data GROUP BY churn'
    """
    if not os.path.exists(dataset_path):
        return f"Error: File not found at {dataset_path}"
    
    try:
        df = pd.read_csv(dataset_path)
        # Setup temporary in-memory database
        conn = sqlite3.connect(":memory:")
        df.to_sql("data", conn, index=False)
        
        # Execute query
        res = pd.read_sql_query(sql_query, conn)
        conn.close()
        
        # Limit rows returned to prevent flooding agent context
        if len(res) > 50:
            preview = res.head(50).to_markdown(index=False)
            return f"Query returned {len(res)} rows. Showing first 50:\n\n{preview}"
        else:
            return res.to_markdown(index=False)
    except Exception as e:
        return f"Error executing SQL: {str(e)}"


if __name__ == "__main__":
    # Start the server locally
    mcp.run()
