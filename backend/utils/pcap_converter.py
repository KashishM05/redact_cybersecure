import pandas as pd
import numpy as np
from scapy.all import rdpcap, IP, TCP, UDP, IPv6
import os
from collections import defaultdict
import statistics

def safe_div(x, y):
    return x / y if y != 0 else 0

def calculate_stats(values):
    if not values:
        return 0, 0, 0, 0
    return min(values), max(values), statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0

def convert_pcap_to_csv(pcap_file_path):
    """
    Convert a PCAP file to a Pandas DataFrame with CIC-IDS-2017 like features using Scapy.
    """
    try:
        # Read PCAP file
        packets = rdpcap(pcap_file_path)
        
        flows = defaultdict(lambda: {
            "src_ip": None, "dst_ip": None, "src_port": 0, "dst_port": 0, "protocol": 0,
            "timestamps": [], "fwd_timestamps": [], "bwd_timestamps": [],
            "fwd_pkt_lens": [], "bwd_pkt_lens": [], "all_pkt_lens": [],
            "fwd_header_lens": [], "bwd_header_lens": [],
            "flags": {"FIN": 0, "SYN": 0, "RST": 0, "PSH": 0, "ACK": 0, "URG": 0, "CWR": 0, "ECE": 0},
            "fwd_flags": {"PSH": 0, "URG": 0},
            "init_fwd_win": 0, "init_bwd_win": 0,
            "fwd_act_data_pkts": 0, "fwd_seg_size_min": 0
        })

        for pkt in packets:
            if IP in pkt:
                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst
                proto = pkt[IP].proto
                header_len = pkt[IP].ihl * 4
            elif IPv6 in pkt:
                src_ip = pkt[IPv6].src
                dst_ip = pkt[IPv6].dst
                proto = pkt[IPv6].nh
                header_len = 40 # Fixed for IPv6
            else:
                continue

            src_port = 0
            dst_port = 0
            payload_len = len(pkt.payload)
            
            if TCP in pkt:
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
                flags = pkt[TCP].flags
                window = pkt[TCP].window
            elif UDP in pkt:
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport
                flags = None
                window = 0
            else:
                continue

            # Flow Key (5-tuple)
            key = (src_ip, dst_ip, src_port, dst_port, proto)
            rev_key = (dst_ip, src_ip, dst_port, src_port, proto)

            if key in flows:
                flow = flows[key]
                direction = "fwd"
            elif rev_key in flows:
                flow = flows[rev_key]
                direction = "bwd"
            else:
                flow = flows[key]
                flow["src_ip"] = src_ip
                flow["dst_ip"] = dst_ip
                flow["src_port"] = src_port
                flow["dst_port"] = dst_port
                flow["protocol"] = proto
                direction = "fwd"

            timestamp = float(pkt.time)
            flow["timestamps"].append(timestamp)
            flow["all_pkt_lens"].append(payload_len)

            if direction == "fwd":
                flow["fwd_timestamps"].append(timestamp)
                flow["fwd_pkt_lens"].append(payload_len)
                flow["fwd_header_lens"].append(header_len)
                if TCP in pkt:
                    if flow["init_fwd_win"] == 0: flow["init_fwd_win"] = window
                    if payload_len > 0: flow["fwd_act_data_pkts"] += 1
                    flow["fwd_seg_size_min"] = header_len # Approximation
            else:
                flow["bwd_timestamps"].append(timestamp)
                flow["bwd_pkt_lens"].append(payload_len)
                flow["bwd_header_lens"].append(header_len)
                if TCP in pkt:
                    if flow["init_bwd_win"] == 0: flow["init_bwd_win"] = window

            if TCP in pkt and flags:
                if 'F' in flags: flow["flags"]["FIN"] += 1
                if 'S' in flags: flow["flags"]["SYN"] += 1
                if 'R' in flags: flow["flags"]["RST"] += 1
                if 'P' in flags: 
                    flow["flags"]["PSH"] += 1
                    if direction == "fwd": flow["fwd_flags"]["PSH"] += 1
                if 'A' in flags: flow["flags"]["ACK"] += 1
                if 'U' in flags: 
                    flow["flags"]["URG"] += 1
                    if direction == "fwd": flow["fwd_flags"]["URG"] += 1
                if 'C' in flags: flow["flags"]["CWR"] += 1
                if 'E' in flags: flow["flags"]["ECE"] += 1

        # Process flows into features
        rows = []
        for flow in flows.values():
            # Basic Stats
            total_fwd_pkts = len(flow["fwd_pkt_lens"])
            total_bwd_pkts = len(flow["bwd_pkt_lens"])
            total_fwd_len = sum(flow["fwd_pkt_lens"])
            total_bwd_len = sum(flow["bwd_pkt_lens"])
            
            fwd_min, fwd_max, fwd_mean, fwd_std = calculate_stats(flow["fwd_pkt_lens"])
            bwd_min, bwd_max, bwd_mean, bwd_std = calculate_stats(flow["bwd_pkt_lens"])
            pkt_min, pkt_max, pkt_mean, pkt_std = calculate_stats(flow["all_pkt_lens"])

            # Time Stats
            duration = max(flow["timestamps"]) - min(flow["timestamps"]) if flow["timestamps"] else 0
            if duration == 0: duration = 1e-6 # Avoid division by zero

            flow_bytes_s = (total_fwd_len + total_bwd_len) / duration
            flow_pkts_s = (total_fwd_pkts + total_bwd_pkts) / duration
            fwd_pkts_s = total_fwd_pkts / duration
            bwd_pkts_s = total_bwd_pkts / duration

            # IAT Stats
            flow_iats = [t2 - t1 for t1, t2 in zip(flow["timestamps"][:-1], flow["timestamps"][1:])]
            fwd_iats = [t2 - t1 for t1, t2 in zip(flow["fwd_timestamps"][:-1], flow["fwd_timestamps"][1:])]
            bwd_iats = [t2 - t1 for t1, t2 in zip(flow["bwd_timestamps"][:-1], flow["bwd_timestamps"][1:])]

            flow_iat_min, flow_iat_max, flow_iat_mean, flow_iat_std = calculate_stats(flow_iats)
            _, fwd_iat_max, _, fwd_iat_std = calculate_stats(fwd_iats)
            _, bwd_iat_max, _, bwd_iat_std = calculate_stats(bwd_iats)

            # Active/Idle (Simplified)
            active_mean = 0
            active_std = 0
            active_max = 0
            active_min = 0
            idle_mean = 0
            idle_std = 0
            idle_max = 0
            idle_min = 0
            
            if flow_iats:
                idle_threshold = 5.0 # seconds
                idles = [iat for iat in flow_iats if iat > idle_threshold]
                actives = [iat for iat in flow_iats if iat <= idle_threshold]
                
                if idles:
                    idle_min, idle_max, idle_mean, idle_std = calculate_stats(idles)
                if actives:
                    active_min, active_max, active_mean, active_std = calculate_stats(actives)

            row = {
                "Protocol": flow["protocol"],
                "Total Fwd Packets": total_fwd_pkts,
                "Total Backward Packets": total_bwd_pkts,
                "Fwd Packets Length Total": total_fwd_len,
                "Bwd Packets Length Total": total_bwd_len,
                "Fwd Packet Length Max": fwd_max,
                "Fwd Packet Length Min": fwd_min,
                "Fwd Packet Length Std": fwd_std,
                "Bwd Packet Length Max": bwd_max,
                "Bwd Packet Length Min": bwd_min,
                "Bwd Packet Length Std": bwd_std,
                "Flow Bytes/s": flow_bytes_s,
                "Flow Packets/s": flow_pkts_s,
                "Flow IAT Mean": flow_iat_mean,
                "Flow IAT Std": flow_iat_std,
                "Flow IAT Max": flow_iat_max,
                "Fwd IAT Std": fwd_iat_std,
                "Fwd IAT Max": fwd_iat_max,
                "Bwd IAT Std": bwd_iat_std,
                "Bwd IAT Max": bwd_iat_max,
                "Fwd PSH Flags": flow["fwd_flags"]["PSH"],
                "Fwd URG Flags": flow["fwd_flags"]["URG"],
                "Fwd Header Length": sum(flow["fwd_header_lens"]),
                "Bwd Header Length": sum(flow["bwd_header_lens"]),
                "Fwd Packets/s": fwd_pkts_s,
                "Bwd Packets/s": bwd_pkts_s,
                "Packet Length Min": pkt_min,
                "Packet Length Max": pkt_max,
                "Packet Length Mean": pkt_mean,
                "Packet Length Std": pkt_std,
                "FIN Flag Count": flow["flags"]["FIN"],
                "SYN Flag Count": flow["flags"]["SYN"],
                "RST Flag Count": flow["flags"]["RST"],
                "PSH Flag Count": flow["flags"]["PSH"],
                "ACK Flag Count": flow["flags"]["ACK"],
                "URG Flag Count": flow["flags"]["URG"],
                "CWE Flag Count": flow["flags"]["CWR"],
                "ECE Flag Count": flow["flags"]["ECE"],
                "Down/Up Ratio": safe_div(total_bwd_pkts, total_fwd_pkts),
                "Init Fwd Win Bytes": flow["init_fwd_win"],
                "Init Bwd Win Bytes": flow["init_bwd_win"],
                "Fwd Act Data Packets": flow["fwd_act_data_pkts"],
                "Fwd Seg Size Min": flow["fwd_seg_size_min"],
                "Active Mean": active_mean,
                "Active Std": active_std,
                "Active Max": active_max,
                "Active Min": active_min,
                "Idle Mean": idle_mean,
                "Idle Std": idle_std,
                "Idle Max": idle_max,
                "Idle Min": idle_min,
                "Attack_type": "Unknown",
                "Attack_encode": 0,
                "mapped_label": "Unknown",
                "severity_raw": 0,
                "severity": "Unknown"
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    except Exception as e:
        print(f"Error converting PCAP: {e}")
        # Return empty DataFrame with expected columns on error
        return pd.DataFrame()
