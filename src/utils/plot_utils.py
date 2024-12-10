#!/usr/bin/python3
# -*- coding: utf-8 -*-

feature_name_d = {
    "duration": "Duration",
    "packet_nb": "Nb pkt",
    "byte_nb": "Nb byte",
    "packet_nb_orig": "Nb pkt from src",
    "packet_nb_resp": "Nb pkt from dst",
    "byte_nb_orig": "Nb byte from src",
    "byte_nb_resp": "Nb byte from dst",
    "resp_orig_packet_ratio": "Nb pkt ratio dst/src",
    "resp_orig_byte_ratio": "Nb byte ratio dst/src",
    "packet_size_abs_mean": "Packet size mean",
    "packet_size_orig_abs_mean": "Packet size mean from src",
    "packet_size_resp_abs_mean": "Packet size mean from dst",
    "packet_size_orig_min": "Min. pkt size from src",
    "packet_size_orig_max": "Max. pkt size from src",
    "a_l3_payload_size_orig_total": "Total size from src",
    "a_l3_payload_size_resp_total": "Total size from dst",
    "a_l3_payload_size_oresp_total": "Total size",
    "a_l3_payload_size_oresp_mean": "Mean size",
    "a_l3_payload_size_stddev": "Size standard deviation",
    "a_l3_payload_size_resp_mean": "Mean size from dst",
    "a_l3_payload_size_resp_max": "Max size from dst",
    "a_l3_payload_size_resp_min": "Min size from dst",
    "a_l3_payload_size_orig_mean": "Mean size from src",
    "a_l3_payload_size_orig_max": "Max size from src",
    "a_l3_payload_size_orig_min": "Min size from src",
    "a_resp_orig_l3_packet_nb_ratio": "Ratio dst/src pkt nb",
    "a_resp_orig_l3_payload_size_ratio": "Ratio dst/src size",
    "a_l4_payload_size_orig_total": "L4 payloads total size from src",
    "a_l4_payload_size_resp_total": "L4 payloads total size from dst",
    "a_ip_packet_size_orig_total": "IP packets total size from src",
    "a_ip_packet_size_resp_total": "IP packets total size from dst",

}


def build_short_feature_name_for_display(feature_name):
    print("plot_utils: build_feature_name_for_display: start")

    print(
        f"plot_utils: build_feature_name_for_display: feature_name: {feature_name}"
    )

    if "l4_payload_size" in feature_name and "nf" in feature_name:
        if "l4_payload_size_ne_with_direction" in feature_name and "nf" in feature_name:
            r = f"L4 payl. size w/o e. w/ d. {feature_name.split('_')[7]}"
        else:
            r = f"Size of L4 payload n°{int(feature_name.split('_')[-1]) + 1}"
    elif ("l4_byte_burst_ne" in feature_name and "origin" not in feature_name
          and "responder" not in feature_name and "nf" in feature_name):
        r = f"L4 byte burst w/o e. {feature_name.split('_')[5]}"
    elif ("a_l4_byte_burst" in feature_name and "origin" not in feature_name
          and "responder" not in feature_name and "nf" in feature_name):
        r = f"L4 byte burst {feature_name.split('_')[5]}"
    elif feature_name == "ip_protocol":
        r = "Protocol"
    elif "ip_protocol" in feature_name:
        r = f"IP protocol {feature_name.split('_')[-1]}"
    elif "l3_payload_size" in feature_name and "nf" in feature_name:
        r = f"Size of packet n°{int(feature_name.split('_')[-1]) + 1}"
    elif "history" in feature_name:
        r = f"History"
    elif "service" in feature_name:
        r = f"Service"
    elif "tcp_flag" in feature_name:
        r = f"Flag {feature_name.split('_')[-1]}"
    else:
        if feature_name in feature_name_d.keys():
            r = feature_name_d[feature_name]
        else:
            r = f"No feature name for {feature_name} => check code"

    print(f"plot_utils: build_feature_name_for_display: r: {r}")

    print("plot_utils: build_feature_name_for_display: end")

    return r
