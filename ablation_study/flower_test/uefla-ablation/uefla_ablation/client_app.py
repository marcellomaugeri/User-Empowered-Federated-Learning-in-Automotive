"""uefla-ablation: A Flower / TensorFlow app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from uefla_ablation.task import load_data, load_model


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, model, x_train, y_train, y_train_binary, x_test, y_test, y_test_binary, epochs, batch_size, verbose
    ):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.y_train_binary = y_train_binary
        self.x_test = x_test
        self.y_test = y_test
        self.y_test_binary = y_test_binary
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            [self.y_train, self.y_train_binary],
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        total_loss, fault_type_loss, fault_detection_loss, fault_type_accuracy, fault_detection_accuracy = self.model.evaluate(self.x_test, [self.y_test, self.y_test_binary], verbose=0)
        return total_loss, len(self.x_test), {"accuracy": fault_type_accuracy}


def client_fn(context: Context):
    # Load model and data
    net = load_model()

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    x_train, y_train, y_train_binary, x_test, y_test, y_test_binary = load_data(partition_id, num_partitions)
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")

    # Return Client instance
    return FlowerClient(
        net, x_train, y_train, y_train_binary, x_test, y_test, y_test_binary, epochs, batch_size, verbose
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
