import random
import string
import csv
import os

output_file = "advanced_url_dataset.csv"
target_size_mb = 200

legit_domains = [
    "google.com", "facebook.com", "amazon.in", "microsoft.com",
    "github.com", "linkedin.com", "wikipedia.org"
]

phishing_keywords = [
    "login", "verify", "secure", "update", "account", "bank", "confirm"
]

shorteners = ["bit.ly", "tinyurl.com", "goo.gl"]
tlds = [".xyz", ".top", ".club", ".online", ".info"]

def rand_str(n=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

def legit_url():
    return f"https://www.{random.choice(legit_domains)}/{rand_str(6)}"

def phishing_url():
    return f"http://{random.choice(phishing_keywords)}-{rand_str(5)}{random.choice(tlds)}/{rand_str(12)}"

def ip_url():
    return f"http://{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}/{rand_str(8)}"

def long_url():
    return f"http://{rand_str(20)}.com/{rand_str(50)}"

def subdomain_url():
    return f"http://{rand_str(5)}.{rand_str(5)}.{random.choice(legit_domains)}/{rand_str(10)}"

def short_url():
    return f"http://{random.choice(shorteners)}/{rand_str(7)}"

def param_url():
    return f"https://www.{random.choice(legit_domains)}/page?id={rand_str(5)}&token={rand_str(10)}"

def ftp_url():
    return f"ftp://{rand_str(6)}.com/file/{rand_str(8)}"

def encoded_url():
    return f"http://{rand_str(5)}.com/%{random.randint(10,99)}%{random.randint(10,99)}/{rand_str(6)}"

def port_url():
    return f"http://{rand_str(6)}.com:{random.randint(1000,9999)}/{rand_str(6)}"

generators = [
    (legit_url, 0),
    (phishing_url, 1),
    (ip_url, 1),
    (long_url, 1),
    (subdomain_url, 1),
    (short_url, 1),
    (param_url, 0),
    (ftp_url, 0),
    (encoded_url, 1),
    (port_url, 1)
]

target_bytes = target_size_mb * 1024 * 1024
with open(output_file, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["url", "label"])

    bytes_written = 0
    while bytes_written < target_bytes:
        batch = []
        for _ in range(10000):
            func, label = random.choice(generators)
            batch.append([func(), label])
        writer.writerows(batch)
        f.flush()
        bytes_written = os.path.getsize(output_file)

print("Dataset created:", output_file)