import pandas as pd
from data.database import engine
from sqlalchemy import text

def get_orders_data():
    query = """
        SELECT 
    o.order_id,
    o.customer_id,
    od.product_id,
    od.discount,
    od.quantity,
    od.unit_price,
    (od.quantity * od.unit_price * (1 - od.discount)) AS total_spend
    FROM 
    orders o
    JOIN 
    order_details od ON o.order_id = od.order_id
    WHERE 
    o.shipped_date IS NOT NULL
    """
    with engine.connect() as connection:
        df = pd.read_sql(query, connection)
    return df

def get_customer_orders_date(customer_id, order_id):
    query = text("""
        SELECT 
        o.customer_id,
            od.discount,
            od.quantity,
            od.unit_price,
            (od.quantity * od.unit_price * (1 - od.discount)) AS total_spend
        FROM orders o 
        JOIN order_details od 
        ON o.order_id = od.order_id
        WHERE o.customer_id = :customer_id AND od.order_id = :order_id
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={
            "customer_id": customer_id,
            "order_id": order_id
        })

    return df


# df = get_customer_orders_date("VINET", 10248)
