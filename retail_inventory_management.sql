-- 1. Creating the Schema for Inventory Control Management

DROP SCHEMA IF EXISTS bravissimo_inventory_control_management CASCADE;
CREATE SCHEMA IF NOT EXISTS bravissimo_inventory_control_management;


-- 2. Creating tables for customer's details in the B2 system
-- In B2 system, Customer Service team is able to find the customers file in database based on the informations below

DROP TABLE IF EXISTS bravissimo_inventory_control_management.user_login;
CREATE TABLE IF NOT EXISTS bravissimo_inventory_control_management.user_login (
	user_reference TEXT PRIMARY KEY,
	email_id TEXT,
    	user_password TEXT,
    	first_name TEXT,
	last_name TEXT,
	postcode TEXT,
	address_line_one TEXT,
	address_line_two TEXT,
	address_line_three TEXT,
	county TEXT,
	country TEXT, 
	phone_number INT,
);



-- 3. Creating a table for supplier or manufacturer 
-- Product team is able to pull up the information regarding the fabrics, threads, underwire for the bra, packaging

DROP TABLE IF EXISTS bravissimo_inventory_control_management.supplier;
CREATE TABLE IF NOT EXISTS bravissimo_inventory_control_management.supplier (
	supplier_id TEXT PRIMARY KEY,
	supplier_name TEXT,
	supplier_product TEXT, -- which product the supplier supply? Specification: Fabrics, Threads, Underwire, Packaging?
	address TEXT,
	phone_number INT,
	email TEXT,
	supplier_contract_start_date DATE,
	supplier_contract_end_date DATE,
);



-- 4. Creating a table for purchase order made by Bravissimo to create the items 
-- Product and Finance team will be able to pull up this information for product research, purchase / restock volume, projection of profit/loss, etc...
-- Referencing / connecting the the this table with the 'supplier_id' and 'supplier_product' from 'supplier' table

DROP TABLE IF EXISTS bravissimo_inventory_control_management.purchase_orders;
CREATE TABLE IF NOT EXISTS bravissimo_inventory_control_management.purchase_orders (
	order_id TEXT PRIMARY KEY,
	supplier_id TEXT REFERENCES bravissimo_inventory_control_management.supplier (supplier_id),
	supplier_product TEXT REFERENCES bravissimo_inventory_control_management.supplier (supplier_product),
	order_date DATE,
	quantity INT,
	delivery_date DATE,
	is_delivered BOOLEAN,
	payment_id TEXT,
	is_paid BOOLEAN
);



-- 5. Creating a table for all items available to buy from Bravissimo
-- Primary key will be the item code
-- All of the product will have different name, colour and size combination, and even different type (bra, swimwear, brief)
-- Side notes #1: NOT ALL item code starts with the bra line name (Eg: FY for Freya, PN for Panache)
-- Side notes #2: Some item such as Bravissimo bra has item code LG or LN, but Bravissimo swimsuit item code are SW or SN
-- The table will be attached with the stock count in inventory, cost per unit and full price per unit

DROP TABLE IF EXISTS bravissimo_inventory_control_management.product_items;
CREATE TABLE IF NOT EXISTS bravissimo_inventory_control_management.product_items (
	item_code TEXT PRIMARY KEY, -- Read side note #1 and #2 
	item_name TEXT,
	item_type TEXT,
	item_size INT, 
	item_colour TEXT,
	item_description TEXT,
	item_image JSON,
	bra_line_name TEXT,
	supplier_id TEXT REFERENCES bravissimo_inventory_control_management.supplier (supplier_id),
	cost_per_unit FLOAT,
	full_price_per_unit FLOAT,
	stock_count INT
);



-- 6. Creating table for when the customer purchased an item from Bravissimo 
-- Primary key is the order_id 
-- Referencing / connecting this table with the 'user_reference' and 'postcode' from 'user_login' table
-- Referencing / connecting the this table with the 'item_code', 'item_colour', and 'item size' from 'product_item' table

DROP TABLE IF EXISTS bravissimo_inventory_control_management.customer_order_purchase;
CREATE TABLE IF NOT EXISTS bravissimo_inventory_control_management.customer_order_purchase (
	order_id INT PRIMARY KEY,
	user_reference TEXT REFERENCES bravissimo_inventory_control_management.user_login (user_reference),
	postcode TEXT REFERENCES bravissimo_inventory_control_management.user_login (postcode),
	date_of_purchase DATE,
	item_code TEXT,
	item_colour TEXT,
	item_size INT,
	is_delivered BOOLEAN,
	payment_id TEXT,
	is_paid BOOLEAN,
	FOREIGN KEY (item_code, item_colour, item_size)
		REFERENCES bravissimo_inventory_control_management.product_items (item_code, item_colour, item_size)
);

