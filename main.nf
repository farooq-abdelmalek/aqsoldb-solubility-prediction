/*
 * AqSolDB reproducible training and inference workflow.
 *
 * Default mode:
 *   Reconstruct the frozen model, then predict LogS.
 *
 * Optional mode:
 *   Supply --model_path to skip training.
 *
 * The held-out test set is never evaluated.
 */

nextflow.enable.dsl = 2

params.input_csv = null
params.data_csv = 'data/curated-solubility-dataset.csv'
params.selection_path = 'results/selected_model.json'
params.model_path = null


process TRAIN_MODEL {

    tag 'frozen-xgboost-training'

    container 'aqsoldb-solubility:dev'

    input:
    path dataset_csv
    path selection_json

    output:
    path 'model.joblib'

    script:
    """
    python -m scripts.train_for_inference \
        --data "${dataset_csv}" \
        --selection "${selection_json}" \
        --output model.joblib
    """
}


process PREDICT_LOGS {

    tag 'molecular-solubility'

    container 'aqsoldb-solubility:dev'

    publishDir 'pipeline_outputs/nextflow',
        mode: 'copy'

    input:
    path molecules_csv
    path trained_model

    output:
    path 'predictions.csv'

    script:
    """
    python -m scripts.predict \
        --input "${molecules_csv}" \
        --model "${trained_model}" \
        --output predictions.csv
    """
}


workflow {

    if (!params.input_csv) {
        error "Provide --input_csv."
    }

    molecules = file(params.input_csv, checkIfExists: true)

    if (params.model_path) {

        // Inference-only mode: reuse an existing trained model.
        model = file(params.model_path, checkIfExists: true)
        PREDICT_LOGS(molecules, model)

    } else {

        // Reproducible mode: reconstruct the frozen model.
        dataset = file(params.data_csv, checkIfExists: true)
        selection = file(params.selection_path, checkIfExists: true)

        TRAIN_MODEL(dataset, selection)

        // Nextflow passes the model produced by TRAIN_MODEL
        // into PREDICT_LOGS automatically.
        PREDICT_LOGS(molecules, TRAIN_MODEL.out)
    }
}
