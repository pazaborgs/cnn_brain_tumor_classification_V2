import mlflow
import pandas as pd
from pathlib import Path

def export_mlflow_runs():
    """Exporta o histórico do MLflow para um CSV estático.
    
    Funciona como fallback de dados de execução (leaderboard) para ambientes
    onde o banco de dados MLflow nativo não está disponível ou é lento (Streamlit Cloud).
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    try:
        all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
        df_runs = mlflow.search_runs(experiment_ids=all_experiments) if all_experiments else pd.DataFrame()
        
        if df_runs.empty:
            print("[AVISO] Nenhuma run encontrada no MLflow local.")
            return

        clean_df = pd.DataFrame()
        if "tags.mlflow.runName" in df_runs.columns:
            clean_df["Model Name"] = df_runs["tags.mlflow.runName"]
        elif "tags.model_name" in df_runs.columns:
            clean_df["Model Name"] = df_runs["tags.model_name"]
        else:
            clean_df["Model Name"] = "Desconhecido"
            
        metrics = [
            "metrics.train_accuracy", "metrics.train_loss",
            "metrics.val_accuracy", "metrics.val_loss", 
            "metrics.test_accuracy", "metrics.test_loss",
            "metrics.macro_f1_score", "metrics.weighted_f1_score"
        ]
        for m in metrics:
            if m in df_runs.columns:
                clean_name = m.replace("metrics.", "").replace("_", " ").title()
                clean_df[clean_name] = df_runs[m]
                
        params = [
            "params.train_batch_size", "params.train_epochs", 
            "params.train_learning_rate", "params.model_base", 
            "params.model_weights", "params.dataset"
        ]
        for p in params:
            if p in df_runs.columns:
                clean_name = p.replace("params.", "").replace("_", " ").title()
                clean_df[clean_name] = df_runs[p]
                
        clean_df["Status"] = df_runs["status"]
        if "end_time" in df_runs.columns:
            clean_df["Data"] = df_runs["end_time"].dt.strftime("%Y-%m-%d %H:%M")

        out_dir = Path("artifacts/metrics")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "mlflow_runs_snapshot.csv"
        
        clean_df.to_csv(out_path, index=False)
        print(f"[SUCESSO] Snapshot exportado: {out_path} ({len(clean_df)} runs)")
        
    except Exception as e:
        print(f"[ERRO] Falha ao exportar snapshot: {e}")

if __name__ == "__main__":
    export_mlflow_runs()
