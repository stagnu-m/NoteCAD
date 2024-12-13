using UnityEngine;
using UnityEngine.UI;

namespace Assets.Code.Tools
{
	public class ChooseBlockPoints : MonoBehaviour
	{
		private static ToggleGroup ToggleGroup;
		// Start is called before the first frame update
		void Start()
		{
			ToggleGroup = GetComponent<ToggleGroup>();
		}

		// Update is called once per frame
		void Update()
		{

		}

		public static bool IsNoBlock()
		{
			Toggle toggle = ToggleGroup.GetFirstActiveToggle();
			if (toggle.name == "No_Block") return true;
			return false;
		}

		public static bool IsProStep()
		{
			Toggle toggle = ToggleGroup.GetFirstActiveToggle();
			if (toggle.name == "Pro_Step") return true;
			return false;
		}

		public static bool IsAfterSolution()
		{
			Toggle toggle = ToggleGroup.GetFirstActiveToggle();
			if (toggle.name == "After_Solution") return true;
			return false;
		}
	}
}