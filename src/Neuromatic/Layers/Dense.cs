using Neuromatic.Initializers;
using System;
using System.Collections.Generic;
using System.Text;
using Neuromatic.Activations;
using TensorFlow;

namespace Neuromatic.Layers
{
    /// <summary>
    /// <para>
    /// Dense layers are used to create fully connected layers in neural networks.
    /// The layer implements the function output = activation(dot(input,kernel) + bias).
    /// </para>
    /// <para>
    /// <b>activation</b> is the element-wise activation function applied to the summed inputs. 
    /// <b>kernel</b> is a set of weights initialized using the weights initialization function.
    /// <b>bias</b> is a vector initialized using the bias initialization function.
    /// </para>
    /// </summary>
    public class Dense : Layer
    {
        private readonly int _units;
        private readonly Layer _input;
        private readonly InitializationFunction _weightsInitializer;
        private readonly InitializationFunction _biasInitializer;
        private readonly ActivationFunction _activation;

        /// <summary>
        /// Initializes a new instance of <see cref="Dense"/>
        /// </summary>
        /// <param name="units">Kernel size of the layer</param>
        /// <param name="activation">Activation function to use (default sigmoid)</param>
        /// <param name="input">Input layer to connect to</param>
        /// <param name="biasInitializer">Initialization function for the bias vector (default random normal)</param>
        /// <param name="weightsInitializer">Initialization function for the weights matrix (default random normal)</param>
        /// <param name="name">Name of the layer</param>
        /// <remarks>
        /// When no name is provided for a layer, one will be generated when you compile the model.
        /// </remarks>
        public Dense(int units, Layer input, ActivationFunction activation = null, InitializationFunction weightsInitializer = null, InitializationFunction biasInitializer = null, string name = null) : base(name)
        {
            if (units <= 0)
            {
                throw new ArgumentException("Invalid layer size. Should be greater than zero", nameof(units));
            }

            _units = units;
            _activation = activation ?? new Sigmoid();
            _weightsInitializer = weightsInitializer ?? new RandomNormal();
            _biasInitializer = biasInitializer ?? new RandomNormal();
            _input = input ?? throw new ArgumentNullException(nameof(input));
        }

        /// <summary>
        /// Gets the output shape for the layer
        /// </summary>
        public override long[] OutputShape
        {
            get
            {
                long[] shape = new long[_input.OutputShape.Length];

                for (int index = 0; index < _input.OutputShape.Length - 1; index++)
                {
                    shape[index] = _input.OutputShape[index];
                }

                shape[shape.Length - 1] = _units;

                return shape;
            }
        }

        /// <summary>
        /// <para>
        /// Builds the layer by converting the abstract definition of the layer into 
        /// a concrete set of instructions for Tensorflow and a layer configuration
        /// for use when training the model.
        /// </para>
        /// <para>This method should register any parameters and initializers with the compilation context.
        /// So that they can be used during the training phase. </para>
        /// <para>Additionally you are required to store the layer configuration in the 
        /// <see cref="context"/> property. This information is required as metadata 
        /// when the model is used.</para>
        /// </summary>
        /// <param name="context">Use this context to register trainable parameters
        /// and build the computational graph for the layer</param>
        public override TFOutput Compile(ModelCompilationContext context)
        {
            if (Configuration != null)
            {
                return Configuration.Output;
            }

            var input = _input.Compile(context);
            var inputDimension = _input.OutputShape[_input.OutputShape.Length - 1];

            using (var scope = context.Graph.WithScope(Name))
            {
                TFShape weightsShape = new TFShape(inputDimension, _units);
                TFShape biasShape = new TFShape(_units);

                var weights = context.Graph.VariableV2(
                    weightsShape,
                    TFDataType.Double, operName: "Weights");

                var bias = context.Graph.VariableV2(
                    biasShape,
                    TFDataType.Double, operName: "Bias");

                var initializers = new[]
                {
                    context.Graph.Assign(weights, _weightsInitializer.Compile(context.Graph, weightsShape)).Operation,
                    context.Graph.Assign(bias, _biasInitializer.Compile(context.Graph, biasShape)).Operation
                };

                // Formula: f(input * W + b)
                // TODO: Make sure that we can work without a bias term.
                var output = _activation.Compile(context, context.Graph.Add(context.Graph.MatMul(input, weights), bias));


                context.AddParameters(weights, bias);
                context.AddInitializers(initializers);

                Configuration = new LayerConfiguration(new[] { weights, bias }, initializers, output);

                return output;
            }
        }
    }
}
