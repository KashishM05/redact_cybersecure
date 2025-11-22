import numpy as np
import pandas as pd
import shap

from merkle_ledger import ThreatLogEntry
from batched_ledger import BatchedThreatLogLedger


ATTACK_RISK_MAP_DATASET_A = {
    "Benign": 0.00,
    "DoS Hulk Attack": 0.80,
    "DDoS Attack": 0.85,
    "DoS GoldenEye Attack": 0.80,
    "DoS slowloris Attack": 0.80,
    "DoS Slowhttptest Attack": 0.80,
    "FTP-Patator Attack": 0.75,
    "SSH-Patator Attack": 0.75,
    "PortScan": 0.60,
    "Web Brute Force Attack": 0.70,
    "Web XSS Attack": 0.65,
    "Web Sql Injection Attack": 0.80,
    "Botnet Activity": 0.90,
    "Infiltration Attack": 0.95,
    "Heartbleed Exploit": 1.00,
}


class DatasetAIntrusionTriageEngine:
    def __init__(
        self,
        model,
        feature_columns,
        batch_size=50,
        benign_label="Benign",
        background_data=None,
    ):
        self.model = model
        self.feature_columns = feature_columns
        self.benign_label = benign_label
        self.ledger = BatchedThreatLogLedger(batch_size=batch_size)
        if background_data is not None:
            self.explainer = shap.TreeExplainer(self.model, data=background_data)
        else:
            self.explainer = shap.TreeExplainer(self.model)

    def _get_risk_factor(self, attack_label):
        return ATTACK_RISK_MAP_DATASET_A.get(attack_label, 0.70)

    def _compute_shap_strength(self, shap_vector):
        if shap_vector is None:
            return 1.0
        mean_abs = float(np.mean(np.abs(shap_vector)))
        return float(np.tanh(mean_abs))

    def _compute_severity(self, attack_label, proba_vec, shap_vector):
        risk_factor = self._get_risk_factor(attack_label)
        if proba_vec is not None:
            model_confidence = float(np.max(proba_vec))
        else:
            model_confidence = 0.8
        shap_strength = self._compute_shap_strength(shap_vector)
        raw = 100.0 * risk_factor * model_confidence * shap_strength
        return int(max(0, min(100, round(raw))))

    def severity_band(score: int) -> str:
        if score <= 30:
            return "Low"
        if score <= 60:
            return "Moderate"
        if score <= 90:
            return "High"
    return "Extremely High"

    def _select_action(self, severity):
        if severity >= 90:
            return "BLOCK"
        if severity >= 75:
            return "QUARANTINE"
        return "ALERT"

    def _prepare_shap_values(self, X):
        shap_values = self.explainer.shap_values(X)
        return shap_values

    def _get_shap_vector_for_sample(self, shap_values_all, class_index, row_index):
        if isinstance(shap_values_all, list):
            if class_index < len(shap_values_all):
                return shap_values_all[class_index][row_index]
            return shap_values_all[0][row_index]
        return shap_values_all[row_index]

    def process_dataframe(self, df, flow_id_col=None):
        X = df[self.feature_columns].values
        y_pred = self.model.predict(X)
        proba = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
        shap_values_all = self._prepare_shap_values(X)
        classes_ = None
        if hasattr(self.model, "classes_"):
            classes_ = list(self.model.classes_)
        results = []
        for idx, row in df.iterrows():
            if flow_id_col is not None and flow_id_col in df.columns:
                flow_id = row[flow_id_col]
            else:
                flow_id = idx
            attack_label = str(y_pred[list(df.index).index(idx)]) if not isinstance(y_pred, pd.Series) else str(y_pred.loc[idx])
            if attack_label == self.benign_label:
                results.append(
                    {
                        "flow_id": flow_id,
                        "is_malicious": False,
                        "attack_label": attack_label,
                        "severity": 0,
                        "action": "NONE",
                    }
                )
                continue
            if proba is not None:
                if classes_ is not None:
                    class_idx = classes_.index(attack_label)
                else:
                    class_idx = int(np.argmax(proba[list(df.index).index(idx)]))
                proba_vec = proba[list(df.index).index(idx)]
            else:
                proba_vec = None
                if classes_ is not None:
                    class_idx = classes_.index(attack_label)
                else:
                    class_idx = 0
            shap_vec = self._get_shap_vector_for_sample(
                shap_values_all,
                class_idx,
                list(df.index).index(idx),
            )
            severity = self._compute_severity(attack_label, proba_vec, shap_vec)
            band = severity_band(severity)
            action = self._select_action(severity)
            entry = ThreatLogEntry.create(
                flow_id=str(flow_id),
                src_ip="",
                dst_ip="",
                attack_label=attack_label,
                severity=severity,
                action=action,
            )
            self.ledger.add_entry(entry)
            results.append(
                {
                    "flow_id": flow_id,
                    "is_malicious": True,
                    "attack_label": attack_label,
                    "severity": severity,
                    "severity_band": band,
                    "action": action,
                }
            )
        return results

    def seal_open_batch(self):
        return self.ledger.seal_current_batch()

    def get_blocks(self):
        return self.ledger.blocks

    def get_ledger_state(self):
        return self.ledger.to_dict()

    def verify_all(self):
        return self.ledger.verify_all()


if __name__ == "__main__":
    model = None
    df_test = None
    feature_columns = []
    engine = DatasetAIntrusionTriageEngine(
        model=model,
        feature_columns=feature_columns,
        batch_size=50,
        benign_label="Benign",
        background_data=None,
    )
    if df_test is not None:
        results = engine.process_dataframe(df_test)
        ledger_state = engine.get_ledger_state()
        all_ok = engine.verify_all()
