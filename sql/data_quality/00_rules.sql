-- Rule catalog. Files 10..40 implement each rule; this table documents them in one place.
-- Upsert so the catalog follows the code on every run.

INSERT INTO dq.rules (rule_id, table_name, category, action, description) VALUES
-- customers
('CUS_DUP_EXACT',          'customers', 'duplicate',           'deduplicate', 'Row identical to an earlier row'),
('CUS_DUP_ID_CONFLICT',    'customers', 'duplicate',           'reject',      'customer_id appears with different contents'),
('CUS_MISSING_ID',         'customers', 'missing',             'reject',      'customer_id is null'),
('CUS_MISSING_COUNTRY',    'customers', 'missing',             'reject',      'Home country is null (needed for every geo feature)'),
('CUS_INVALID_VALUE',      'customers', 'invalid',             'reject',      'Unknown country or segment, or unparseable date'),
('CUS_UNDERAGE_ACCOUNT',   'customers', 'inconsistent',        'reject',      'Account opened before the customer turned 18'),
-- cards
('CRD_DUP_EXACT',          'cards',     'duplicate',           'deduplicate', 'Row identical to an earlier row'),
('CRD_DUP_ID_CONFLICT',    'cards',     'duplicate',           'reject',      'card_id appears with different contents'),
('CRD_INVALID_VALUE',      'cards',     'invalid',             'reject',      'Missing id, unknown card type or unparseable date'),
('CRD_ORPHAN_CUSTOMER',    'cards',     'broken_relationship', 'reject',      'customer_id does not exist in the customer extract'),
('CRD_PARENT_REJECTED',    'cards',     'broken_relationship', 'reject',      'Customer exists but was rejected by another rule'),
('CRD_ISSUED_BEFORE_ACCOUNT', 'cards',  'inconsistent',        'reject',      'Card issued before the account was opened'),
-- merchants
('MER_DUP_EXACT',          'merchants', 'duplicate',           'deduplicate', 'Row identical to an earlier row'),
('MER_DUP_ID_CONFLICT',    'merchants', 'duplicate',           'reject',      'merchant_id appears with different contents'),
('MER_INVALID_VALUE',      'merchants', 'invalid',             'reject',      'Missing id, unknown category/channel/risk/country or bad date'),
-- devices
('DEV_DUP_EXACT',          'devices',   'duplicate',           'deduplicate', 'Row identical to an earlier row'),
('DEV_DUP_ID_CONFLICT',    'devices',   'duplicate',           'reject',      'device_id appears with different contents'),
('DEV_INVALID_VALUE',      'devices',   'invalid',             'reject',      'Missing id, unknown type or country, or unparseable date'),
-- transactions
('TXN_DUP_EXACT',          'transactions', 'duplicate',        'deduplicate', 'Row identical to an earlier row (retried message)'),
('TXN_DUP_ID_CONFLICT',    'transactions', 'duplicate',        'reject',      'transaction_id appears with different amount/time: no version can be trusted'),
('TXN_MISSING_ID',         'transactions', 'missing',          'reject',      'transaction_id is null'),
('TXN_MISSING_CUSTOMER',   'transactions', 'missing',          'repair',      'customer_id is null: recovered from the card owner (a card has exactly one owner)'),
('TXN_INVALID_AMOUNT',     'transactions', 'invalid',          'reject',      'Amount null, not numeric or <= 0'),
('TXN_CURRENCY_FORMAT',    'transactions', 'invalid',          'repair',      'Valid ISO currency in wrong case/whitespace: normalised'),
('TXN_INVALID_CURRENCY',   'transactions', 'invalid',          'reject',      'Currency is not a known ISO code'),
('TXN_INVALID_TIMESTAMP',  'transactions', 'invalid',          'reject',      'Timestamp null or unparseable'),
('TXN_FUTURE_TIMESTAMP',   'transactions', 'inconsistent',     'reject',      'Timestamp after the extraction time'),
('TXN_BEFORE_CARD_ISSUE',  'transactions', 'inconsistent',     'reject',      'Timestamp before the card was issued'),
('TXN_INVALID_VALUE',      'transactions', 'invalid',          'reject',      'Unknown payment method or country'),
('TXN_UNKNOWN_CUSTOMER',   'transactions', 'broken_relationship', 'reject',   'customer_id does not exist in the customer extract'),
('TXN_UNKNOWN_CARD',       'transactions', 'broken_relationship', 'reject',   'card_id does not exist in the card extract'),
('TXN_UNKNOWN_MERCHANT',   'transactions', 'broken_relationship', 'reject',   'merchant_id does not exist in the merchant extract'),
('TXN_UNKNOWN_DEVICE',     'transactions', 'broken_relationship', 'reject',   'device_id does not exist in the device extract'),
('TXN_CARD_CUSTOMER_MISMATCH', 'transactions', 'broken_relationship', 'reject', 'Card belongs to a different customer'),
('TXN_PARENT_REJECTED',    'transactions', 'broken_relationship', 'reject',   'Customer, card, merchant or device was rejected by another rule'),
('TXN_SENTINEL_DEVICE',    'transactions', 'invalid',          'repair',      'Placeholder device id ("unknown", "00000000"): device set to NULL'),
('TXN_DEVICE_ON_CARD_PRESENT', 'transactions', 'inconsistent', 'reject',      'Card-present payment carrying a customer device'),
('TXN_COUNTRY_MISMATCH',   'transactions', 'inconsistent',     'repair',      'Card-present payment whose country differs from the merchant country: terminal country = merchant country'),
('TXN_NO_FX_RATE',         'transactions', 'broken_relationship', 'reject',   'No exchange rate for the currency on that day'),
('TXN_AMOUNT_OUTLIER',     'transactions', 'outlier',          'warn',        'Amount above the 99.9th percentile of its merchant category (kept: fraud lives in the tail)'),
('TXN_SHARED_DEVICE',      'transactions', 'outlier',          'warn',        'Device used by more than 50 distinct customers in the extract'),
-- fraud labels
('LBL_DUP_EXACT',          'fraud_labels', 'duplicate',        'deduplicate', 'Row identical to an earlier row'),
('LBL_DUP_ID_CONFLICT',    'fraud_labels', 'duplicate',        'reject',      'Two different labels for the same transaction'),
('LBL_INVALID_VALUE',      'fraud_labels', 'invalid',          'reject',      'is_fraud not boolean, unknown source or unparseable report date'),
('LBL_UNKNOWN_TRANSACTION','fraud_labels', 'broken_relationship', 'reject',   'Label for a transaction that is not in the extract'),
('LBL_PARENT_REJECTED',    'fraud_labels', 'broken_relationship', 'reject',   'Labelled transaction was rejected by another rule'),
('LBL_REPORTED_BEFORE_TXN','fraud_labels', 'inconsistent',     'reject',      'Label reported before the transaction happened')
ON CONFLICT (rule_id) DO UPDATE SET
    table_name  = EXCLUDED.table_name,
    category    = EXCLUDED.category,
    action      = EXCLUDED.action,
    description = EXCLUDED.description;
