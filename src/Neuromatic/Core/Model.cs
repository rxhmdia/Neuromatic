using Neuromatic.Core;
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
        public IEnumerable<Layer> Inputs { get; }

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
            foreach (var input in Inputs)
            {
                input.Compile(backend);
            }

            foreach (var output in Outputs)
            {
                output.Compile(backend);
            }

            foreach(var (output, loss) in Enumerable.Zip(Outputs, Losses, (output,loss) => (output, loss)))
            {
                loss.Compile(output, backend);
            }

            foreach(var (output, metric) in Enumerable.Zip(Outputs, Metrics, (output, metric) => (output, metric)))
            {
                metric.Compile(output, backend);
            }

            Optimizer.Compile(backend);
            
            return backend.Output;
        }
    }
}
