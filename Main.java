import java.util.*;

public class Main {

    public static void main(String[] args){

        String test = "abcabcbb";
        int GetMax = lengthOfLongestSubstring(test);
        System.out.println(GetMax);

    }

    public static int lengthOfLongestSubstring(String s) 
    {
        HashMap<Character, Integer> tracker = new HashMap<>();
        int maxlength = 0;
        int left = 0;

        for (int right = 0; right < s.length(); right++)
        {
            char current = s.charAt(right);

            if(tracker.containsKey(current))
            {
                left = Math.max(tracker.get(current)+1, left);
            }
            tracker.put(current, right);
            maxlength = Math.max(maxlength, right - left + 1);


        }
        return maxlength;


        
    }

 

    





    
}
