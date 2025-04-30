import pandas as pd
from data.database import engine
def get_customer_order_data():
    query = """
    SELECT 
        c.customer_id,
        o.order_id,
        o.order_date,
        od.unit_price,
        od.quantity
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_details od ON o.order_id = od.order_id
    """
    with engine.connect() as connection:
        df = pd.read_sql(query, connection)
    return df