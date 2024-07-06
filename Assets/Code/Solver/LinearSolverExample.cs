using System;
using Google.OrTools.Init;
using LinearSolver = Google.OrTools.LinearSolver.Solver;

//test 2


namespace Assets.Code.Solver
{
    public class LinearSolverExample
    {
        private const bool _debug = true;

        public static void SolveLinearProgram(double[,] A, double[] B, ref double[] X)
        {
            Console.WriteLine("Google.OrTools version: " + OrToolsVersion.VersionString());

            // Create the linear solver with the GLOP backend.
            LinearSolver solver = LinearSolver.CreateSolver("GLOP");

            if (solver is null)
            {
                Console.WriteLine("Could not create solver GLOP");
                return;
            }

            var numberVariables = A.GetLength(0);
            var numberConstraints = B.GetLength(0);

            if (_debug)
            {
                Console.WriteLine($"Number of variables: {numberVariables}");
                Console.WriteLine($"Number of constraints: {numberConstraints}");
            }
        }
    }
}