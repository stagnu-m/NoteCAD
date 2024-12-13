using System;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Assets.Code.Solver;
using Assets.Code.Tools;
using Assets.Code.Utils;

public class EquationSystem  {

	public enum SolveResult {
		DEFAULT,
		OKAY,
		DIDNT_CONVEGE,
		REDUNDANT,
		POSTPONE,
		BREAK,
		ITERATION
	}

	bool isDirty = true;

	public bool IsDirty { get { return isDirty; } }
	public int maxSteps = 20;
	public int dragSteps = 3;
	public bool revertWhenNotConverged = true;
	private bool IsNoBlock => ChooseBlockPoints.IsNoBlock();
	private bool IsProStep => ChooseBlockPoints.IsProStep();
	private bool IsAfterSolution => ChooseBlockPoints.IsAfterSolution();
	private bool IsReset = false;
	

	Exp[,] J;
	double[,] A;
	double[,] AAT;
	double[] B;
	double[] X;
	double[] Z;
	double[] oldParamValues;

	List<Exp> sourceEquations = new List<Exp>();
	List<Param> parameters = new List<Param>();

	List<Exp> equations = new List<Exp>();
	List<Param> currentParams = new List<Param>();

	Dictionary<Param, Param> subs;

	public void AddEquation(Exp eq) {
		sourceEquations.Add(eq);
		isDirty = true;
	}

	public void AddEquation(ExpVector v) {
		sourceEquations.Add(v.x);
		sourceEquations.Add(v.y);
		sourceEquations.Add(v.z);
		isDirty = true;
	}

	public void AddEquations(IEnumerable<Exp> eq) {
		sourceEquations.AddRange(eq);
		isDirty = true;
	}

	public void RemoveEquation(Exp eq) {
		sourceEquations.Remove(eq);
		isDirty = true;
	}

	public void AddParameter(Param p) {
		parameters.Add(p);
		isDirty = true;
	}

	public void AddParameters(IEnumerable<Param> p) {
		parameters.AddRange(p);
		isDirty = true;
	}

	public void RemoveParameter(Param p) {
		parameters.Remove(p);
		isDirty = true;
	}

	public void Eval(ref double[] B_, ref List<Exp> equations_) {
		for(int i = 0; i < equations_.Count; i++) {
			//if(clearDrag && equations_[i].IsDrag()) {
			//	B_[i] = 0.0;
			//	continue;
			//}
			B_[i] = equations_[i].Eval();
		}
	}

	public bool IsConverged(ref double[] B_, bool checkDrag, ref List<Exp> equations_, bool printNonConverged = false) {
		for(int i = 0; i < equations_.Count; i++) {
			if(!checkDrag && equations_[i].IsDrag()) {
				continue;
			}
			if(Math.Abs(B_[i]) < GaussianMethod.epsilon) continue;	
			if(printNonConverged) {
				Debug.Log("Not converged: " + equations_[i].ToString());
			}
			return false;
		}
		return true;
	}

	void StoreParams(ref double[] oldParamValues_, List<Param> parameters_) {
		for(int i = 0; i < parameters_.Count; i++) {
			oldParamValues_[i] = parameters_[i].value;
		}
	}

	void RevertParams(ref double[] oldParamValues_, ref List<Param> parameters_) {
		for(int i = 0; i < parameters_.Count; i++) {
			parameters_[i].value = oldParamValues_[i];
		}
	}

	static Exp[,] WriteJacobian(ref List<Exp> equations_, ref List<Param> parameters_) {
		var J = new Exp[equations_.Count, parameters_.Count];
		for(int r = 0; r < equations_.Count; r++) {
			var eq = equations_[r];
			for(int c = 0; c < parameters_.Count; c++) {
				var u = parameters_[c];
				J[r, c] = eq.Deriv(u);
				/*
				if(!J[r, c].IsZeroConst()) {
					Debug.Log(J[r, c].ToString() + "\n");
				}
				*/
			}
		}
		return J;
	}

	public bool HasDragged() {
		return equations.Any(e => e.IsDrag());
	}

	public void EvalJacobian(
		ref List<Exp> equations_,
		ref List<Param> currentParams_,
		ref List<Param> parameters_,
		ref Dictionary<Param, Param> subs_,
		ref Exp[,] J_,
		ref double[,] A_,
		ref double[] B_,
		ref double[] X_,
		ref double[] Z_,
		ref double[,] AAT_,
		ref double[] oldParamValues_,
		bool clearDrag) {
		UpdateDirty(
			ref equations_,
			ref currentParams_,
			ref parameters_,
			ref subs_,
			ref J_,
			ref A_,
			ref B_,
			ref X_,
			ref Z_,
			ref AAT_,
			ref oldParamValues_);	
		UnityEngine.Profiling.Profiler.BeginSample("EvalJacobian");
		for(int r = 0; r < J_.GetLength(0); r++) {
			if(clearDrag && equations_[r].IsDrag()) {
				for(int c = 0; c < J_.GetLength(1); c++) {
					A_[r, c] = 0.0;
				}
				continue;
			}
			for(int c = 0; c < J_.GetLength(1); c++) {
				A_[r, c] = J_[r, c].Eval();
			}
		}
		UnityEngine.Profiling.Profiler.EndSample();
	}

	public void SolveLeastSquares(double[,] A, double[] B, ref double[] X) {

		// A^T * A * X = A^T * B
		var rows = A.GetLength(0);
		var cols = A.GetLength(1);

		UnityEngine.Profiling.Profiler.BeginSample("SolveLeastSquares: A^T * A");
		var time = Time.realtimeSinceStartup;
		for(int r = 0; r < rows; r++) {
			for(int c = 0; c < rows; c++) {
				double sum = 0.0;
				for(int i = 0; i < cols; i++) {
					if(A[c, i] == 0 || A[r, i] == 0) continue;
					sum += A[r, i] * A[c, i];
				}
				AAT[r, c] = sum;
			}
		}
		//Debug.Log("AAT time " + (Time.realtimeSinceStartup - time) * 1000);
		UnityEngine.Profiling.Profiler.EndSample();

		GaussianMethod.Solve(AAT, B, ref Z);

		for(int c = 0; c < cols; c++) {
			double sum = 0.0;
			for(int r = 0; r < rows; r++) {
				sum += Z[r] * A[r, c];
			}
			X[c] = sum;
		}

	}

	public void Clear() {
		parameters.Clear();
		currentParams.Clear();
		equations.Clear();
		sourceEquations.Clear();
		isDirty = true;
		UpdateDirty(
			ref equations,
			ref currentParams,
			ref parameters,
			ref subs,
			ref J,
			ref A,
			ref B,
			ref X,
			ref Z,
			ref AAT,
			ref oldParamValues);
	}

	public bool TestRank(out int dof) {
		EvalJacobian(
			ref equations,
			ref currentParams,
			ref parameters,
			ref subs,
			ref J,
			ref A,
			ref B,
			ref X,
			ref Z,
			ref AAT,
			ref oldParamValues,
			clearDrag: false);
		int rank = GaussianMethod.Rank(A);
		dof = A.GetLength(1) - rank;
		return rank == A.GetLength(0);
	}

	public int GetDoF(double[,] matrix)
	{
		int rank = GaussianMethod.Rank(matrix);
		int dof = matrix.GetLength(1) - rank;
		return dof;
	}

	void UpdateDirty(ref List<Exp> equations_,
	                 ref List<Param> currentParams_,
	                 ref List<Param> parameters_,
	                 ref Dictionary<Param, Param> subs_,
	                 ref Exp[,] J_,
	                 ref double[,] A_,
	                 ref double[] B_,
	                 ref double[] X_,
	                 ref double[] Z_,
	                 ref double[,] AAT_,
	                 ref double[] oldParamValues_
		) {
		if(isDirty) {
			equations_ = sourceEquations.Select(e => e.DeepClone()).ToList();
			currentParams_ = parameters_.ToList();
			/*
			foreach(var e in equations) {
				e.ReduceParams(currentParams);
			}*/
			//currentParams = parameters.Where(p => equations.Any(e => e.IsDependOn(p))).ToList();
			subs_ = SolveBySubstitution(ref equations_, ref currentParams_);

			J_ = WriteJacobian(ref equations_, ref currentParams_);
			A_ = new double[J_.GetLength(0), J_.GetLength(1)];
			B_ = new double[equations_.Count];
			X_ = new double[currentParams_.Count];
			Z_ = new double[A_.GetLength(0)];
			AAT_ = new double[A_.GetLength(0), A_.GetLength(0)];
			oldParamValues_ = new double[parameters_.Count];
			isDirty = false;
			dofChanged = true;
		}
	}

	void BackSubstitution(ref Dictionary<Param, Param> subs_, ref List<Param> parameters_) {
		if(subs_ == null) return;
		for(int i = 0; i < parameters_.Count; i++) {
			var p = parameters_[i];
			if(!subs_.ContainsKey(p)) continue;
			p.value = subs_[p].value;
		}
	}

	Dictionary<Param, Param> SolveBySubstitution(
		ref List<Exp> equations_,
		ref List<Param> currentParams_) {
		var subs_ = new Dictionary<Param, Param>();

		for(int i = 0; i < equations_.Count; i++) {
			var eq = equations_[i];
			if(!eq.IsSubstitionForm()) continue;
			var a = eq.GetSubstitutionParamA();
			var b = eq.GetSubstitutionParamB();
			if(Math.Abs(a.value - b.value) > GaussianMethod.epsilon) continue;
			if(!currentParams_.Contains(b)) {
				var t = a;
				a = b;
				b = t;
			}
			// TODO: Check errors
			//if(!parameters.Contains(b)) {
			//	continue;
			//}

			foreach(var k in subs_.Keys.ToList()) {
				if(subs_[k] == b) {
					subs_[k] = a;
				}
			}
			subs_[b] = a;
			equations_.RemoveAt(i--);
			currentParams_.Remove(b);

			for(int j = 0; j < equations_.Count; j++) {
				equations_[j].Substitute(b, a);
			}
		}
		return subs_;
	}

	public string stats { get; private set; }
	public bool dofChanged { get; private set; }

	public SolveResult Solve() {
		dofChanged = false;
		IsReset = false;
		UpdateDirty(
			ref equations,
			ref currentParams,
			ref parameters,
			ref subs,
			ref J,
			ref A,
			ref B,
			ref X,
			ref Z,
			ref AAT,
			ref oldParamValues);
		StoreParams(ref oldParamValues, parameters);

		

		int steps = 0;
		int counter = 0;

		var dof = 0;
		bool isInitial = true;
		var initialSolution = new double[X.Length];
		var initialParamPosition = new double[X.Length];
		List<IndexToBlock> indicesToConsider = new List<IndexToBlock>();

		var dragIndices = new List<int>();

		for (int i = 0; i < currentParams.Count; i++)
		{
			initialParamPosition[i] = currentParams[i].value;
			if (!currentParams[i].IsDrag) continue;
			var currentParamIndex = currentParams.FindIndex(x => x == currentParams[i]);
			dragIndices.Add(currentParamIndex);
		}

		//-----------------------------------------------------------------------------------------------------------------------
		// If IsNoBlock
		if (IsNoBlock)
		{
			var empty_solution = new double[X.Length];
			var defaultSolver = DefaultSolver(steps, initialSolution, initialParamPosition, indicesToConsider, dragIndices, null, empty_solution, false, dof);
			if (defaultSolver == SolveResult.OKAY)
			{
				return defaultSolver;
			}
		}

		//-----------------------------------------------------------------------------------------------------------------------
		// If IsAfterSolution
		if (IsAfterSolution)
		{
					do
					{
						//IndexToBlock indexToBlock = null;
						var indexToBlock = indicesToConsider.FirstOrDefault(x => !x.IsUsed);
						//if (!isInitial)
						//{
						//var indexToBlock = new IndexToBlock { Index_x = 4, Index_y = 5 };
						//}

						steps = 0;
						var completeSolution = new double[X.Length];
						do
						{
							var result = SolveResult.DEFAULT;
							if (isInitial)
							{
								result = SolveResultL1(
									ref steps,
									ref initialSolution,
									ref initialParamPosition,
									ref indicesToConsider,
									ref dragIndices,
									ref indexToBlock,
									ref completeSolution,
									ref isInitial,
									ref equations,
									ref subs,
									ref parameters,
									ref currentParams,
									ref J,
									ref A,
									ref B,
									ref X,
									ref Z,
									ref AAT,
									ref oldParamValues,
									ref dof);
							}

							if (result == SolveResult.BREAK)
							{
								counter++;
								break;
							}

							if (!isInitial)
							{
								if (counter == 1)
								{
									isDirty = true;
									UpdateDirty(
										ref equations,
										ref currentParams,
										ref parameters,
										ref subs,
										ref J,
										ref A,
										ref B,
										ref X,
										ref Z,
										ref AAT,
										ref oldParamValues);
									StoreParams(ref oldParamValues, parameters);
									counter = 2;
								}
								result = SolveResultL1(
									ref steps,
									ref initialSolution,
									ref initialParamPosition,
									ref indicesToConsider,
									ref dragIndices,
									ref indexToBlock,
									ref completeSolution,
									ref isInitial,
									ref equations,
									ref subs,
									ref parameters,
									ref currentParams,
									ref J,
									ref A,
									ref B,
									ref X,
									ref Z,
									ref AAT,
									ref oldParamValues,
									ref dof);
							}

							if (result == SolveResult.ITERATION) continue;
							if (result == SolveResult.OKAY)
							{
								return result;
							}
						} while(steps++ <= maxSteps);
					} while (indicesToConsider.Count(x => !x.IsUsed) > 0);

					steps = 0;
					var empty_solution = new double[X.Length];

					isDirty = true;
					RevertParams(ref oldParamValues, ref parameters);
					UpdateDirty(
						ref equations,
						ref currentParams,
						ref parameters,
						ref subs,
						ref J,
						ref A,
						ref B,
						ref X,
						ref Z,
						ref AAT,
						ref oldParamValues);
					StoreParams(ref oldParamValues, parameters);

					var defaultSolver = DefaultSolver(steps, initialSolution, initialParamPosition, indicesToConsider, dragIndices, null, empty_solution, false, dof);
					if (defaultSolver == SolveResult.OKAY)
					{
						return defaultSolver;
					}
		}

		//-----------------------------------------------------------------------------------------------------------------------
		// If IsProStep

		if (IsProStep)
		{
			var empty_solution = new double[X.Length];
			var defaultSolver = DefaultSolver(steps, initialSolution, initialParamPosition, indicesToConsider, dragIndices, null, empty_solution, false, dof);
			if (defaultSolver == SolveResult.OKAY)
			{
				return defaultSolver;
			}

			// if no solution found, use initial one
			IsReset = true;
			isDirty = true;
			RevertParams(ref oldParamValues, ref parameters);
			UpdateDirty(
				ref equations,
				ref currentParams,
				ref parameters,
				ref subs,
				ref J,
				ref A,
				ref B,
				ref X,
				ref Z,
				ref AAT,
				ref oldParamValues);
			StoreParams(ref oldParamValues, parameters);

			defaultSolver = DefaultSolver(steps, initialSolution, initialParamPosition, indicesToConsider, dragIndices, null, empty_solution, false, dof);
			if (defaultSolver == SolveResult.OKAY)
			{
				return defaultSolver;
			}
		}

		//var notConverge = IsConverged(checkDrag: false, printNonConverged: true);
		//Debug.Log($"DIDNT_CONVEGE is {notConverge}");
		IsConverged(ref B,checkDrag: true, ref equations, printNonConverged: true);
		if (revertWhenNotConverged) {
			RevertParams(ref oldParamValues, ref parameters);
			dofChanged = false;
		}
		return SolveResult.DIDNT_CONVEGE;
	}

	private SolveResult DefaultSolver(
		int steps, 
		double[] initialSolution, 
		double[] initialParamPosition,
		List<IndexToBlock> indicesToConsider, 
		List<int> dragIndices, 
		IndexToBlock nullIndex,
		double[] empty_solution, 
		bool isInitial, 
		int dof)
	{
		SolveResult result;
		do
		{
			result = SolveResult.DEFAULT;

			result = SolveResultL1(
				ref steps,
				ref initialSolution,
				ref initialParamPosition,
				ref indicesToConsider,
				ref dragIndices,
				ref nullIndex,
				ref empty_solution,
				ref isInitial,
				ref equations,
				ref subs,
				ref parameters,
				ref currentParams,
				ref J,
				ref A,
				ref B,
				ref X,
				ref Z,
				ref AAT,
				ref oldParamValues,
				ref dof);

			if (result == SolveResult.ITERATION) continue;
			if (result == SolveResult.OKAY)
			{
				return result;
			}
		} while (steps++ <= maxSteps);

		return result;
	}

	private SolveResult SolveResultL1(ref int steps,
	                                    ref double[] initialSolution,
	                                    ref double[] initialParamPosition,
	                                    ref List<IndexToBlock> indicesToConsider,
	                                    ref List<int> dragIndices,
	                                    ref IndexToBlock indexToBlock,
	                                    ref double[] completeSolution,
	                                    ref bool isInitial,
	                                    ref List<Exp> equations_,
	                                    ref Dictionary<Param, Param> subs_,
	                                    ref List<Param> params_,
	                                    ref List<Param> currentParams_,
	                                    ref Exp[,] J_,
	                                    ref double[,] A_,
	                                    ref double[] B_,
	                                    ref double[] X_,
	                                    ref double[] Z_,
	                                    ref double[,] AAT_,
	                                    ref double[] oldParamValues_,
	                                    ref int dof)
	{
		Eval(ref B_, ref equations_);

		if (IsConverged(ref B_, checkDrag: true, ref equations_))
		{
			if (isInitial && steps > 0)
			{
				BlockFreeCoordinates(completeSolution, dragIndices, dof, indicesToConsider);
				isInitial = false;

				RevertParams(ref oldParamValues_, ref params_);
				//indicesToConsider.Add(new IndexToBlock { Index_x = 4, Index_y = 5 });
				//indicesToConsider.Add(new IndexToBlock { Index_x = 0, Index_y = 1 });
				//indicesToConsider.Add(new IndexToBlock { Index_x = 4, Index_y = 5 });
				return SolveResult.BREAK;
			}

			if (steps > 0)
			{
				dofChanged = true;
				Debug.Log(
					$"solved {equations_.Count} equations with {currentParams_.Count} unknowns in {steps} steps");
			}
			stats = $"eqs:{equations_.Count}\nunkn: {currentParams_.Count}";
			BackSubstitution(ref subs_, ref params_);
			return SolveResult.OKAY;
		}

		EvalJacobian(
			ref equations_,
			ref currentParams_,
			ref params_,
			ref subs_,
			ref J_,
			ref A_,
			ref B_,
			ref X_,
			ref Z_,
			ref AAT_,
			ref oldParamValues_,
			clearDrag: false);

		// TODO rewrite to solve for l_1
		
		var copyA = DeepCopyMatrix(A_);
		dof = GetDoF(copyA);
		Debug.Log($"Degree of Freedom is: {dof} \n");
		if (IsProStep && !IsReset)
		{
			LinearSolverExample.SolveLinearProgram(A_, B_, ref X_, dragIndices, dof);
		}
		else
		{
			LinearSolverExample.SolveLinearProgramInitial(ref A_, ref B_, ref X_, dragIndices, dof, indexToBlock);
		}

		for (int i = 0; i < currentParams_.Count; i++)
		{
			currentParams_[i].value -= X_[i];
			completeSolution[i] -= X_[i];
		}
		return SolveResult.ITERATION;
	}

	#region Helper functions

    private static void BlockFreeCoordinates(double[] X, List<int> dragIndices, int dof, List<IndexToBlock> indicesToConsider)
    {
	    // Number of points in X (each point has two components)
	    int numPairs = X.Length / 2;

	    // If no DOF, exit early
	    if (dof == 0) return;

	    if (indicesToConsider.Count >= dof) return;

	    // Iterate over all pairs (points) in X
	    for (int i = 0; i < numPairs; i++)
	    {
		    // Skip drag indices
		    int varIndex_x = 2 * i;
		    int varIndex_y = 2 * i + 1;
		    if (dragIndices.Contains(varIndex_x) || dragIndices.Contains(varIndex_y)) continue;

		    // Current point components
		    double x_coordinate = X[varIndex_x];
		    double y_coordinate = X[varIndex_y];

		    // Skip if both coordinates are zero or both are non-zero
		    if (x_coordinate == 0 ^ y_coordinate == 0)
		    {
			    //TODO
			    indicesToConsider.Add(new IndexToBlock
			    {
				    Index_x = varIndex_x,
				    Value_x = x_coordinate,
				    Index_y = varIndex_y,
				    Value_y = y_coordinate,
				    IsUsed = false
			    });
		    }
			indicesToConsider.Sort((a, b) => a.AbsoluteSum.CompareTo(b.AbsoluteSum));
		}

		//indicesToConsider = indicesToConsider.OrderBy(x => x.AbsoluteSum).ToList();

	}

    // Create a method to print the 2D array
    string MatrixToString(double[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int columns = matrix.GetLength(1);
        string result = "";

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                result += matrix[i, j].ToString("F2") + "\t"; // Format with 2 decimal points
            }
            result += "\n";
        }

        return result;
    }

    private static double[,] DeepCopyMatrix(double[,] original)
    {
        int rows = original.GetLength(0);
        int cols = original.GetLength(1);
        double[,] copy = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                copy[i, j] = original[i, j];
            }
        }
        return copy;
    }

    private static double[] DeepCopyArray(double[] original)
    {
        double[] copy = new double[original.Length];

        for (int i = 0; i < original.Length; i++)
        {
            copy[i] = original[i];
        }
        return copy;
    }

    Dictionary<Param, Param> DeepCopyDictionary(Dictionary<Param, Param> original)
    {
	    var deepCopy = new Dictionary<Param, Param>();

	    foreach (var kvp in original)
	    {
		    // Deep clone both key and value
		    var clonedKey = CloneUtility.DeepClone(kvp.Key);
		    var clonedValue = CloneUtility.DeepClone(kvp.Value);
		    deepCopy[clonedKey] = clonedValue;
	    }

	    return deepCopy;
    }


    #endregion

}
