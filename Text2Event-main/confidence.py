# Collaborative Filtering for Large and Small Models
# Input: Samples generated by the large model and samples predicted by the small model (confidence needs to be manually copied from the predicted samples)
# Output: Consistent samples, high-confidence small model predictions with inconsistencies, low-confidence large model generations, different outputs from large and small model annotations
# python confidence.py

import json

# import json

def compare_events(event1, event2):
    if len(event1["events_info"]) != len(event2["events_info"]):
        return False
    for e1, e2 in zip(event1["events_info"], event2["events_info"]):
        if e1.get("trigger_text") != e2.get("trigger_text"):
            return False
        if e1.get("event_type") != e2.get("event_type"):
            return False
        args1 = e1.get("argument", [])
        args2 = e2.get("argument", [])
        if len(args1) != len(args2):
            return False
        for a1, a2 in zip(args1, args2):
            if a1.get("text") != a2.get("text"):
                return False
            if a1.get("role") != a2.get("role"):
                return False
    return True
    
def process_data(data_list):
    processed_data = []
    for data in data_list:
        if "confidence" in data:
            del data["confidence"]
        processed_data.append(data)
    return processed_data
# Data generated by a large model, data predicted by a small model
def main(real_data_file, predicted_data_file):
    with open(real_data_file, "r", encoding="utf-8") as f1:
        real_data = json.load(f1)
        confidence_values = [0.0485, 0.0868, 0.2075, 0.0064, 0.0181, 0.0968, 0.1526, 0.2524, 0.0107, 0.0276, 0.1769, 0.0113, 0.0415, 0.0868, 0.1913, 0.0072, 0.0113, 0.2192, 0.0158, 0.0069, 0.0083, 0.7564, 0.0263, 0.0047, 0.1079, 0.0074, 0.0092, 0.0197, 0.0089, 0.0157, 0.0321, 0.0463, 0.4199, 0.1797, 0.0111, 0.0817, 0.4473, 0.0062, 0.01, 0.0301, 0.0085, 0.02, 0.0535, 0.0868, 0.1378, 0.0109, 0.0147, 0.0063, 0.1726, 0.0081, 0.0145, 0.0532, 0.4288, 0.208, 0.0353, 0.0471, 0.0419, 0.0868, 0.1304, 0.0071, 0.0162, 0.1663, 0.4161, 0.0676, 0.0016, 0.111, 0.2733, 0.0165, 0.0224, 0.0607, 0.0868, 0.1304, 0.0164, 0.0128, 0.571, 0.0707, 0.009, 0.0339, 0.0089, 0.2235, 0.0116, 0.0621, 0.06, 0.0063, 0.01, 0.0444, 0.0744, 0.0093, 0.1383, 0.014, 0.1746, 0.0136, 0.0155, 0.0275, 0.1571, 0.0111, 0.8675, 0.0078, 0.0144, 0.0612, 0.0592, 0.0414, 0.0868, 0.1304, 0.0097, 0.022, 0.6505, 0.1976, 0.1759, 0.0544, 0.1735, 0.4011, 0.2466, 0.1969, 0.0557, 0.0868, 0.1499, 0.0258, 0.0292, 0.0616, 0.0095, 0.0039, 0.0085, 0.0051, 0.336, 0.0069, 0.1293, 0.0046, 0.0489, 0.0868, 0.1369, 0.0066, 0.0225, 0.0109, 0.0039, 0.0041, 0.0066, 0.0099, 0.017, 0.0426, 0.0235, 0.0067, 0.0024, 0.0084, 0.0064, 0.0076, 0.0165, 0.0105, 0.0093, 0.1379, 0.0095, 0.3052, 0.0196, 0.0043, 0.007, 0.0085, 0.0089, 0.0041, 0.0207, 0.0309, 0.0868, 0.0807, 0.0106, 0.0102, 0.2841, 0.3644, 0.6777, 0.1402, 0.2109, 0.9725, 0.442, 0.0806, 0.0083, 0.0442, 0.0868, 0.0718, 0.0051, 0.1399, 0.1171, 0.0045, 0.1629, 0.3705, 0.0087, 0.0179, 0.1159, 0.1437, 0.0144, 0.0133, 0.0158, 0.0368, 0.0098, 0.0192, 0.1571, 0.0114, 0.0157, 0.0082, 0.0103, 0.0407, 0.0088, 0.0072, 0.0092, 0.0137, 0.0071, 0.0063, 0.122, 0.0081, 0.0524, 0.0868, 0.1318, 0.0077, 0.0194, 0.0797, 0.076, 0.0601, 0.0142, 0.0457, 0.0102, 0.0112, 0.0142, 0.3876, 0.0408, 0.0868, 0.1613, 0.0286, 0.0255, 0.0597, 0.0789, 0.0384, 0.3288, 0.0296, 0.0228, 0.0105, 0.0219, 0.0232, 0.0138, 0.0173, 0.0243, 0.0193, 0.0091, 0.0069, 0.0359, 0.0185, 0.0657, 0.0868, 0.2656, 0.0108, 0.029, 0.0934, 0.1766, 0.0745, 0.0129, 0.0848, 0.015, 0.1673, 0.1135, 0.0095, 0.0147, 0.0124, 0.0426, 0.0868, 0.2309, 0.0062, 0.0093, 0.0987, 0.0192, 0.1931, 0.2998, 0.1469, 0.0374, 0.0868, 0.1429, 0.0223, 0.0151, 0.2006, 0.1231, 0.0577, 0.0213, 0.1553, 0.0106, 0.1432, 0.0288, 0.0868, 0.1429, 0.0133, 0.0174, 0.1385, 0.1717, 0.1292, 0.0291, 0.0387, 0.0191, 0.3265, 0.1001, 0.1515, 0.4849, 0.0171, 0.0438, 0.0868, 0.174, 0.0072, 0.0158, 0.0321, 0.0083, 0.0098, 0.1789, 0.1272, 0.0072, 0.0147, 0.0079, 0.0195, 0.0154, 0.0113, 0.0239, 0.0631, 0.0868, 0.0919, 0.0163, 0.0172, 0.0777, 0.1456, 0.0431, 0.0502, 0.0249, 0.0182, 0.1355, 0.0181, 0.1226, 0.0441, 0.0868, 0.176, 0.0058, 0.0191, 0.0086, 0.3089, 0.1015, 0.3337, 0.0095, 0.0136, 0.1339, 0.1253, 0.1199, 0.0566, 0.0868, 0.176, 0.0055, 0.0216, 0.0037, 0.0083, 0.2936, 0.0102, 0.1035, 0.0065, 0.0029, 0.1956, 0.0862, 0.1149, 0.022, 0.0039, 0.013, 0.0216, 0.0044, 0.0086, 0.0093, 0.0363, 0.091, 0.444, 0.3692, 0.031, 0.7792, 0.1983, 0.5345, 0.5583, 0.0061, 0.0155, 0.0074, 0.0127, 0.0117, 0.0067, 0.048, 0.0868, 0.1988, 0.0093, 0.0118, 0.1203, 0.0905, 0.145, 0.2446, 0.0128, 0.0972, 0.1503, 0.0183, 0.2974, 0.1311, 0.0052, 0.0121, 0.0083, 0.0286, 0.0092, 0.0107, 0.0172, 0.0339, 0.0265, 0.0229, 0.0128, 0.8675, 0.0075, 0.0175, 0.0192, 0.056, 0.0868, 0.0839, 0.031, 0.0197, 0.0678, 0.0751, 0.0076, 0.0147, 0.241, 0.0513, 0.0347, 0.0124, 0.0613, 0.0868, 0.0606, 0.0288, 0.0145, 0.1808, 0.2675, 0.0062, 0.2451, 0.013, 0.0471, 0.0868, 0.1722, 0.0164, 0.0279, 0.0084, 0.0021, 0.0114, 0.0066, 0.5424, 0.459, 0.0347, 0.0868, 0.1815, 0.0144, 0.0979, 0.2501, 0.0268, 0.1576, 0.2175, 0.1326, 0.0054, 0.0135, 0.1475, 0.0234, 0.0096, 0.0497, 0.0233, 0.0179, 0.013, 0.016, 0.0056, 0.0146, 0.0127, 0.016, 0.0791, 0.174, 0.0069, 0.0106, 0.1571, 0.0215, 0.0056, 0.2663, 0.0985, 0.0324, 0.0868, 0.1526, 0.0162, 0.0205, 0.0111, 0.0058, 0.0124, 0.0129, 0.0142, 0.0089, 0.5388, 0.0091, 0.0494, 0.0868, 0.1192, 0.0054, 0.0101, 0.202, 0.0504, 0.3049, 0.0843, 0.3727, 0.4207, 0.0073, 0.1419, 0.0074, 0.0396, 0.0868, 0.1005, 0.0101, 0.0302, 0.1385, 0.0997, 0.0097, 0.0124, 0.062, 0.043, 0.2244, 0.0055, 0.0436, 0.0868, 0.1219, 0.0179, 0.0105, 0.0238, 0.0145, 0.1142, 0.0137, 0.0143, 0.02, 0.1749, 0.0403, 0.0868, 0.1266, 0.0099, 0.0144, 0.0655, 0.2656, 0.0101, 0.2771, 0.5306, 0.1406, 0.2193, 0.3367, 0.138, 0.0043, 0.0058, 0.01, 0.0171, 0.007, 0.0163, 0.0071, 0.0136, 0.01, 0.227, 0.0129, 0.07, 0.0383, 0.0868, 0.2337, 0.0162, 0.0217, 0.4197, 0.0356, 0.0245, 0.4372, 0.006, 0.1541, 0.0062, 0.0133, 0.1386, 0.0343, 0.2459, 0.5127, 0.0096, 0.0091, 0.0076, 0.0071, 0.0109, 0.011, 0.0151, 0.0234, 0.0184, 0.0131, 0.0249, 0.0514, 0.0177, 0.012, 0.0531, 0.0868, 0.2553, 0.0044, 0.0174, 0.2479, 0.0667, 0.2875, 0.2432, 0.006, 0.1174, 0.0072, 0.2277, 0.0419, 0.0868, 0.1966, 0.0118, 0.025, 0.0039, 0.0011, 0.0027, 0.0101, 0.0132, 0.0743, 0.0844, 0.0056, 0.1922, 0.1645, 0.0074, 0.123, 0.0091, 0.0015, 0.0061, 0.0101, 0.0183, 0.0448, 0.0097, 0.0081, 0.009, 0.0045, 0.0048, 0.0105, 0.0066, 0.0166, 0.0101, 0.3625, 0.0081, 0.0318, 0.0868, 0.2109, 0.0292, 0.0159, 0.009, 0.0069, 0.0186, 0.0055, 0.0216, 0.0046, 0.0977, 0.0076, 0.0179, 0.0298, 0.0868, 0.2609, 0.009, 0.022, 0.1373, 0.1681, 0.0162, 0.1774, 0.036, 0.021, 0.0242, 0.0085, 0.006, 0.0134, 0.032, 0.0167, 0.1674, 0.0332, 0.0149, 0.1011, 0.2655, 0.2451, 0.2146, 0.0119, 0.1886, 0.0072, 0.1894, 0.012, 0.5001, 0.3007, 0.0175, 0.1391, 0.0143, 0.516, 0.0211, 0.0119, 0.0076, 0.0287, 0.0059, 0.4925, 0.0481, 0.0868, 0.2102, 0.0072, 0.0185, 0.2348, 0.1101, 0.1459, 0.0743, 0.2473, 0.0121, 0.0359, 0.1182, 0.071, 0.1721, 0.017, 0.0069, 0.0181, 0.0114, 0.0066, 0.0171, 0.1098, 0.0488, 0.0157, 0.1317, 0.2031, 0.2328, 0.0592, 0.0407, 0.111, 0.1202, 0.0098, 0.1151, 0.1833, 0.156, 0.1102, 0.156, 0.3677, 0.0107, 0.113, 0.0404, 0.0868, 0.2544, 0.0113, 0.039, 0.2499, 0.1041, 0.136, 0.0565, 0.0105, 0.0143, 0.0157, 0.0132, 0.0201, 0.028, 0.0188, 0.12, 0.113, 0.0222, 0.0705, 0.0119, 0.311, 0.0069, 0.0366, 0.0868, 0.2323, 0.0025, 0.0292, 0.3335, 0.0042, 0.7516, 0.2429, 0.1788, 0.1664, 0.1595, 0.0086, 0.2426, 0.156, 0.0019, 0.156, 0.0065, 0.1778, 0.2359, 0.0721, 0.1027, 0.1102, 0.0072, 0.0914, 0.0067, 0.1272, 0.0027, 0.0166, 0.1123, 0.2627, 0.3948, 0.2236, 0.0059, 0.447, 0.1599, 0.008, 0.5485, 0.2539, 0.0775, 0.0086, 0.0075, 0.0768, 0.7703, 0.0755, 0.136, 0.031, 0.0051, 0.0339, 0.0868, 0.0942, 0.0159, 0.0318, 0.1344, 0.3802, 0.0177, 0.0154, 0.0237, 0.0188, 0.3558, 0.018, 0.009, 0.0122, 0.2741, 0.0154, 0.0416, 0.0868, 0.1838, 0.0086, 0.0256, 0.1485, 0.1145, 0.005, 0.0745, 0.0448, 0.188, 0.0204, 0.1776, 0.3123, 0.0117, 0.1522, 0.0134, 0.0083]
        
        for i in range(len(real_data)):
            real_data[i]['confidence'] = confidence_values[i]# 真实数据加上置信度
    with open('gold.json', 'w') as f:
        json.dump(real_data, f, indent=4)
    with open(predicted_data_file, "r", encoding="utf-8") as f2:
        predicted_data = json.load(f2)

        

        for i in range(len(predicted_data)):
            predicted_data[i]['confidence'] = confidence_values[i]

    with open('pred.json', 'w') as f:
        json.dump(predicted_data, f, indent=4)
    # Ensure that real_data and predicted_data have the same length
    assert len(real_data) == len(predicted_data)
    same_data = []
    confident_data = []
    low_confident_data = []
    unconfident_data_SLM = []
    unconfident_data_LLM = []
    different_data_LLM = []
    different_data_SLM = []
    for i in range(len(real_data)):
        events_info1 = real_data[i]["events_info"]
        events_info2 = predicted_data[i]["events_info"]
    
        matched = compare_events(real_data[i], predicted_data[i])
    
        if matched:
            same_data.append(real_data[i])
        else:
            different_data_LLM.append(real_data[i])
            different_data_SLM.append(predicted_data[i])

            # Check confidence
            confidence = predicted_data[i].get("confidence", 0)
            if confidence > 0.95:
                confident_data.append(predicted_data[i])
            elif confidence <= 0.05:
                low_confident_data.append(real_data[i])
            else:
                unconfident_data_SLM.append(predicted_data[i])
                unconfident_data_LLM.append(real_data[i])



# Merge and process all data
    all_data = same_data + confident_data + low_confident_data
    data_process = process_data(all_data)
    unconfident_data_SLM = process_data(unconfident_data_SLM)
    unconfident_data_LLM = process_data(unconfident_data_LLM)
    

    with open("processed_data/same_data.json", "w", encoding="utf-8") as f:
        json.dump(same_data, f, ensure_ascii=False, indent=4)
    with open("processed_data/different_data_LLM.json", "w", encoding="utf-8") as f:
        json.dump(different_data_LLM, f, ensure_ascii=False, indent=4)
    with open("processed_data/different_data_SLM.json", "w", encoding="utf-8") as f:
        json.dump(different_data_SLM, f, ensure_ascii=False, indent=4)

    with open("processed_data/high_confident.json", "w", encoding="utf-8") as f:
        json.dump(confident_data, f, ensure_ascii=False, indent=4)

    with open("processed_data/low_confident_LLM.json", "w", encoding="utf-8") as f:
        json.dump(low_confident_data, f, ensure_ascii=False, indent=4)

    with open("processed_data/unconfident_data_SLM.json", "w", encoding="utf-8") as f:
        json.dump(unconfident_data_SLM, f, ensure_ascii=False, indent=4)
    with open("processed_data/unconfident_data_LLM.json", "w", encoding="utf-8") as f:
        json.dump(unconfident_data_LLM, f, ensure_ascii=False, indent=4)
    with open("processed_data/processed_data1.json", "w", encoding="utf-8") as f:
        json.dump(data_process, f, ensure_ascii=False, indent=4)


main("processDATA/data.json", "processDATA/updated_data.json")

