using Neuromatic.Layers;
using Neuromatic.Losses;
using Neuromatic.Metrics;
using Neuromatic.Optimizers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Neuromatic.Core
{
    /// <summary>
    /// Defines a model with a set of inputs and outputs
    /// </summary>
    public class Model
    {

        /// <summary>
        /// Initializes a new instance of <see cref="Model"/>
        /// </summary>
        /// <param name="inputs">Inputs for the model</param>
        /// <param name="outputs">Outputs for the model</param>
        public Model(IEnumerable<Input> inputs, IEnumerable<Layer> outputs, IEnumerable<LossFunction> losses, Optimizer optimizer, IEnumerable<MetricFunction> metrics)
        {
            Inputs = inputs;
            Outputs = outputs;
            Losses = losses;
            Metrics = metrics;
            Optimizer = optimizer;
        }

        /// <summary>
        /// Gets the input layers for the model
        /// </summary>
        public IEnumerable<Input> Inputs { get; }

        /// <summary>
        /// Gets the output layers for the model
        /// </summary>
        public IEnumerable<Layer> Outputs { get; }

        /// <summary>
        /// Gets the loss functions for each of the model outputs
        /// </summary>
        public IEnumerable<LossFunction> Losses { get; }

        /// <summary>
        /// Gets the metrics for the model
        /// </summary>
        public IEnumerable<MetricFunction> Metrics { get; }

        /// <summary>
        /// Gets the optimizer for the model
        /// </summary>
        public Optimizer Optimizer { get; }

        /// <summary>
        /// Compiles the model using the provided model visitor to create the executable model.
        /// You need to provide a specific visitor to compile the model such as the TensorFlowModelVisitor.
        /// </summary>
        /// <param name="backend">Backend used to compile the model</param>
        /// <returns>Returns an executable model</returns>
        public ExecutableModel Compile(ModelBackend backend)
        {
            var inputs = CompileInputs(backend);
            var outputs = CompileOutputs(backend);
            var targets = CompileTargets(backend);
            var losses = CompileLossFunctions(outputs, targets, backend);
            var modelLoss = CompileModelLoss(losses, backend);

            foreach (var (output, metric) in Enumerable.Zip(Outputs, Metrics, (output, metric) => (output, metric)))
            {
                metric.Compile(output, backend);
            }

            var optimizer = Optimizer.Compile(modelLoss, backend.TrainableWeights, backend);
            var trainingFunction = CompileTrainingFunction(inputs, outputs, targets, losses, optimizer, backend);
            var predictFunction = CompilePredictionFunction(inputs, outputs, backend);

            return new ExecutableModel(trainingFunction, predictFunction, inputs, outputs, targets, losses);
        }

        /// <summary>
        /// Compiles the inputs for the model
        /// </summary>
        /// <param name="backend"></param>
        /// <returns></returns>
        IEnumerable<ExecutableModelNode> CompileInputs(ModelBackend backend)
        {
            return Inputs.Select(x => x.Compile(backend)).ToList();
        }

        /// <summary>
        /// When multiple losses are provided, this function combines them by adding up all the values for the loss functions.
        /// When only a single loss function is provided, this returns that loss function instead.
        /// </summary>
        /// <param name="losses">A collection of loss functions defined for the model</param>
        /// <param name="backend">Backend to use for compiling the total loss function</param>
        /// <returns>Returns the compiled model loss function</returns>
        ExecutableModelNode CompileModelLoss(IEnumerable<ExecutableModelNode> losses, ModelBackend backend)
        {
            if (Losses.Count() == 1)
            {
                return losses.Single();
            }

            return losses.Aggregate((left, right) => backend.Add(left, right));
        }

        /// <summary>
        /// Compiles the individual loss functions for the model
        /// </summary>
        /// <param name="backend">Backend to use for compilation</param>
        /// <returns>Returns the compiled loss functions</returns>
        IEnumerable<ExecutableModelNode> CompileLossFunctions(
            IEnumerable<ExecutableModelNode> outputs, 
            IEnumerable<ExecutableModelNode> targets, 
            ModelBackend backend)
        {
            var functionInputs = Enumerable.Zip(outputs,targets, (output, target) => (output, target));

            return Enumerable.Zip(functionInputs, Losses, 
                (input, function) => function.Compile(input.output, input.target, backend)).ToList();
        }

        /// <summary>
        /// Compiles the output nodes for the model and returns the native nodes for them
        /// </summary>
        /// <param name="backend">Backend to use for compilation</param>
        /// <returns>Returns the list of compiled model outputs</returns>
        IEnumerable<ExecutableModelNode> CompileOutputs(ModelBackend backend)
        {
            return Outputs.Select(x => x.Compile(backend)).ToList();
        }

        /// <summary>
        /// Compiles a set of target value placeholders to be used by the loss functions.
        /// The placeholders have the same shape as the outputs.
        /// </summary>
        /// <param name="backend">Backend to use for compilation</param>
        /// <returns>Returns a list of target value placeholders</returns>
        IEnumerable<ExecutableModelNode> CompileTargets(ModelBackend backend)
        {
            return Outputs.Select(x => backend.Placeholder($"{x.Name}_Target", x.Shape)).ToList();
        }

        /// <summary>
        /// Compiles the training function for the model
        /// </summary>
        /// <param name="inputs">List of inputs to process</param>
        /// <param name="outputs">List of outputs to process</param>
        /// <param name="targets">List of targets to process</param>
        /// <param name="losses">List of loss functions to process</param>
        /// <param name="optimizer">Optimizer to use</param>
        /// <param name="backend">Backend to use for compilation</param>
        /// <returns></returns>
        BackendFunction CompileTrainingFunction(
            IEnumerable<ExecutableModelNode> inputs,
            IEnumerable<ExecutableModelNode> outputs,
            IEnumerable<ExecutableModelNode> targets,
            IEnumerable<ExecutableModelNode> losses,
            IEnumerable<ExecutableModelNode> optimizer,
            ModelBackend backend)
        {
            return backend.Function(inputs.Concat(targets), outputs.Concat(losses), optimizer);
        }

        /// <summary>
        /// Compiles the prediction function for the model
        /// </summary>
        /// <param name="inputs">Inputs for the function</param>
        /// <param name="outputs">Outputs for the function</param>
        /// <param name="backend"></param>
        /// <returns></returns>
        BackendFunction CompilePredictionFunction(IEnumerable<ExecutableModelNode> inputs, IEnumerable<ExecutableModelNode> outputs, ModelBackend backend)
        {
            return backend.Function(inputs, outputs, Enumerable.Empty<ExecutableModelNode>());
        }
    }
}
