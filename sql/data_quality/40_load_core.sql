-- Replace core with the clean data of this run. Runs inside the same transaction as the rules:
-- if any constraint fails, nothing is replaced. The core constraints are the last line of
-- defence: a row that slips through a rule fails here and aborts the run loudly.

TRUNCATE core.fraud_labels, core.transactions, core.devices, core.cards,
         core.merchants, core.customers;

INSERT INTO core.customers (customer_id, date_of_birth, country_code, account_created_at,
                            customer_segment)
SELECT customer_id, date_of_birth, country_code, account_created_at, customer_segment
FROM dq.clean_customers;

INSERT INTO core.cards (card_id, customer_id, card_type, issued_at)
SELECT card_id, customer_id, card_type, issued_at
FROM dq.clean_cards;

INSERT INTO core.merchants (merchant_id, merchant_category, channel, country_code, risk_level,
                            onboarded_at)
SELECT merchant_id, merchant_category, channel, country_code, risk_level, onboarded_at
FROM dq.clean_merchants;

INSERT INTO core.devices (device_id, device_type, operating_system, country_code, first_seen_at)
SELECT device_id, device_type, operating_system, country_code, first_seen_at
FROM dq.clean_devices;

-- Inserted in time order: keeps the table physically sorted by ts, which is what makes the
-- BRIN index on ts effective.
INSERT INTO core.transactions (transaction_id, customer_id, card_id, merchant_id, device_id, ts,
                               amount, currency_code, amount_eur, country_code, payment_method)
SELECT transaction_id, customer_id, card_id, merchant_id, device_id, ts,
       amount, currency_code, amount_eur, country_code, payment_method
FROM dq.clean_transactions
ORDER BY ts, transaction_id;

INSERT INTO core.fraud_labels (transaction_id, is_fraud, reported_at, label_source)
SELECT transaction_id, is_fraud, reported_at, label_source
FROM dq.clean_fraud_labels;
