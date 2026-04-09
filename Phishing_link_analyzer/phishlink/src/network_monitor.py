import subprocess
import json
import datetime
import sys
import os

def capture_and_analyze(interface: str, target_domains: set, duration: int):
    """
    Captures network traffic on the specified interface for `duration` seconds.
    Analyzes HTTP Host and TLS SNI to find accesses to `target_domains`.
    If `target_domains` is empty, intercepts ALL unique accessed domains.
    """
    detected_events = []

    # Default paths for TShark
    if sys.platform == 'win32':
        tshark_path = r"C:\Program Files\Wireshark\tshark.exe"
    else:
        tshark_path = "tshark"
        
    if not os.path.exists(tshark_path) and sys.platform == 'win32':
        print(f"Warning: {tshark_path} not found. Ensure Wireshark is installed.")
        # Fallback to expecting it in PATH
        tshark_path = "tshark"

    # We build a subprocess TShark command that outputs JSON directly.
    # We only care about HTTP Host and TLS Server Name.
    cmd = [
        tshark_path, 
        "-i", interface, 
        "-a", f"duration:{duration}", 
        "-T", "json",
        "-e", "ip.src", 
        "-e", "ip.dst", 
        "-e", "frame.time_epoch", 
        "-e", "http.host", 
        "-e", "tls.handshake.extensions_server_name",
        "-Y", "http.request or tls.handshake.type == 1"
    ]
    
    try:
        # Run tshark synchronously. It blocks the Streamlit thread cleanly for `duration` seconds.
        # This completely side-steps any Streamlit + Pyshark 'asyncio' deadlocks.
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        # Output might be empty if 0 packets captured
        output = result.stdout.strip()
        if not output:
            return detected_events
            
        packets = json.loads(output)
        
        for p in packets:
            layers = p.get("_source", {}).get("layers", {})
            
            src_ip = layers.get("ip.src", [""])[0]
            dst_ip = layers.get("ip.dst", [""])[0]
            
            # Epoch format e.g., "1672531199.123"
            epoch_str = layers.get("frame.time_epoch", ["0"])[0]
            try:
                timestamp = datetime.datetime.fromtimestamp(float(epoch_str))
            except ValueError:
                timestamp = datetime.datetime.now()
            
            domain_accessed = None
            if "http.host" in layers:
                domain_accessed = layers["http.host"][0]
            elif "tls.handshake.extensions_server_name" in layers:
                domain_accessed = layers["tls.handshake.extensions_server_name"][0]
                
            if domain_accessed:
                domain_accessed = str(domain_accessed).lower().strip()
                if not target_domains:
                    event = {
                        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "source_ip": src_ip,
                        "destination_ip": dst_ip,
                        "domain": domain_accessed,
                        "matched_target": "N/A (AI Live Scan)"
                    }
                    if event not in detected_events:
                        detected_events.append(event)
                else:
                    for target in target_domains:
                        target_clean = target.lower().replace("http://", "").replace("https://", "").split("/")[0]
                        if target_clean in domain_accessed or domain_accessed in target_clean:
                            event = {
                                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                "source_ip": src_ip,
                                "destination_ip": dst_ip,
                                "domain": domain_accessed,
                                "matched_target": target
                            }
                            if event not in detected_events:
                                detected_events.append(event)
                                
    except FileNotFoundError:
        print("TShark executable was not found. Please install Wireshark and add it to PATH.")
    except json.JSONDecodeError as e:
        print(f"Error parsing TShark JSON output: {e}")
    except Exception as e:
        print(f"Unexpected error capturing packets via subprocess: {e}")

    return detected_events
