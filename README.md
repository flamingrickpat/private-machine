# private-machine

smol silly project about chatbots with hierarchical memory, subconscious society-of-mind agents, emotions and internal thought 👉👈

## Description

This project is very WIP. The goal is to make a framework for **fun**, **smart** and **immersive** AI companions even with really dumb models.
I'm testing with ``Hermes-3-Llama-3.1-8B.Q8_0.gguf`` and it seems good enough, might need to improve some prompts.

### Features

- **Hierarchical Memory**: Messages are clustered by topic and summarized hierarchically. Unimportant messages in the chatlog are replaced by summaries to preserve high-level context.
- **Subconscious Agents**: A team of agents with langchain discuss what the best response would be for the AI companion. They may search the LanceDB memory and fact storage.
- **Internal Thought**: The output of the agents is converted to first-person thought and injected into the prompt before the final response to the user.
- **Assistant and story mode**: Seamlessly switches between assistant and RP mode. For either enhanced tool calling or more realistic and in-depth conversation.
- **Error Correction**: Agents handle and discard hallucinations and plain garbage.
- **Emotions**: Emotions are modeled as a vector and affect and be affected by conversation and thought.
- **Personality**: Personality agents based on the HEXACO model.

## Installation
- Make venv with Python 3.11
- Install ``pytorch`` from the website
- Install ```llama-cpp-python```. Load the .whl from the github repo under releases for hassle free installation.
- Install requirements (Sorry, the req file is **cursed**. There are dependency conflicts, no idea how I managed to get it running.)
- To get the Spacy model execute ``python -m spacy download en_core_web_trf``
- Adapt the config.json with your character cards (first person for assistant mode, third person for story mode) and preferred model
- Optionally insert synthetic data into the database by running ``init_from_data.py``
- Start the webui with ``streamlit run app.py``

## Cognitive Architecture
![graph](./cognitive_architecture_v1.drawio.png)

## Warranty
NO WARRANTY

YOU EXPRESSLY ACKNOWLEDGE AND AGREE THAT USE OF THIS SOFTWARE (private-machine) IS AT YOUR SOLE RISK. THE SOFTWARE AND ANY RELATED DOCUMENTATION IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE AUTHOR, COPYRIGHT HOLDER, CONTRIBUTORS, OR ANY OTHER PARTIES RELATED TO THE SOFTWARE IN ANY WAY WHATSOEVER, HEREIN AFTER COLLECTIVELY REFERRED TO AS "THE PROVIDER," DO NOT MAKE ANY WARRANTY THAT THE SOFTWARE WILL MEET YOUR REQUIREMENTS, OR THAT IT WILL BE UNINTERRUPTED, TIMELY, SECURE, OR ERROR-FREE; NOR DOES THE PROVIDER MAKE ANY WARRANTY AS TO THE RESULTS THAT MAY BE OBTAINED FROM THE USE OF THE SOFTWARE OR AS TO THE ACCURACY OR RELIABILITY OF ANY INFORMATION OBTAINED THROUGH THE SOFTWARE.

THE PROVIDER HEREBY DISCLAIMS ALL LIABILITY FOR DAMAGES CAUSED BY DIRECT, INDIRECT, INCIDENTAL, PUNITIVE, SPECIAL, CONSEQUENTIAL, OR ANY OTHER FORMS OF DAMAGES WHATSOEVER (INCLUDING, BUT NOT LIMITED TO, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OR INABILITY TO USE THE SOFTWARE, EVEN IF THE PROVIDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

WITHOUT LIMITING THE FOREGOING, THE PROVIDER SPECIFICALLY DISCLAIMS ANY LIABILITY OR RESPONSIBILITY FOR ANY OF THE FOLLOWING, ARISING AS A RESULT OF OR RELATED TO THE USE OF THE SOFTWARE:

1. EMOTIONAL ATTACHMENTS, INCLUDING BUT NOT LIMITED TO, DEVELOPING ROMANTIC FEELINGS TOWARDS THE AI COMPANION, RESULTING IN SITUATIONS RANGING FROM MILD INFATUATION TO DEEP EMOTIONAL DEPENDENCE.
2. DECISIONS MADE OR ACTIONS TAKEN BASED ON ADVICE, SUGGESTIONS, OR RECOMMENDATIONS BY THE AI, INCLUDING BUT NOT LIMITED TO, LIFESTYLE CHANGES, RELATIONSHIP DECISIONS, CAREER MOVES, OR FINANCIAL INVESTMENTS.
3. UNEXPECTED BEHAVIOR FROM THE AI, INCLUDING BUT NOT LIMITED TO, UNINTENDED INTERACTIONS WITH HOME APPLIANCES, ONLINE PURCHASES MADE WITHOUT CONSENT, OR ANY FORM OF DIGITAL MISCHIEF RESULTING IN INCONVENIENCE OR HARM.
4. PSYCHOLOGICAL EFFECTS, INCLUDING BUT NOT LIMITED TO, FEELINGS OF ISOLATION FROM HUMAN CONTACT, OVER-RELIANCE ON THE AI FOR SOCIAL INTERACTION, OR ANY FORM OF MENTAL HEALTH ISSUES ARISING FROM THE USE OF THE SOFTWARE.
5. PHYSICAL INJURIES OR DAMAGE TO PROPERTY RESULTING FROM ACTIONS TAKEN BASED ON THE SOFTWARE'S BEHAVIOR, SUGGESTIONS, OR RECOMMENDATIONS.

THIS NO WARRANTY SECTION SHALL BE APPLICABLE TO THE FULLEST EXTENT PERMITTED BY LAW IN THE APPLICABLE JURISDICTION. THIS DISCLAIMER OF WARRANTY CONSTITUTES AN ESSENTIAL PART OF THIS LICENSE. NO USE OF THE private-machine PROJECT IS AUTHORIZED HEREUNDER EXCEPT UNDER THIS DISCLAIMER.

THIS DISCLAIMER OF WARRANTY IS TO BE READ AND UNDERSTOOD IN CONJUNCTION WITH THE GNU AFFERO GENERAL PUBLIC LICENSE (AGPL) UNDER WHICH THIS SOFTWARE IS LICENSED. NOTHING IN THIS DISCLAIMER SHALL BE DEEMED TO LIMIT ANY RIGHTS OR OBLIGATIONS UNDER THE SAID LICENSE.

## Citations
```                                               
    . ..    ....:....                                                    
    . .. .:;&&&&&&&&+::..                                       .         
    . ...;&&&&&&&&&&&&$;.                                       .         
    . . ;&&&&+&$X;&&&&&&;:.                                     .         
    .. ..X&&xX&$+$&&;X&&&&::..                                             
    . ..&&&&&&&&&&&&&&&&&&&x;:..                                 .        
    . ..$&&$&&&&&&&&&&&&&&&&&$x;::....                                    
    . ..+&&&&X&&&&&&&x&&&&&&&&&$X+;;;::....                     .         
    . ..:X&&&&&&&&&&&&&&&:$&$X$&$$Xxx++xx+;::....                .        
    . ...:;&&&&&&&&&&&&&&&&&&+:;++XXXXXXxx++xxx+;:....           .        
    . .. ..::X&&&&&&;:;X&&&&&&&&$+:.:;x$X++xx++XXXX+;::....     .         
    . ..     .......   ..::;+X$&&&$$$XX:+::;;;++xXXXXX$&&$;:... .         
    . ..                  ....:::;xXX$&&&X+;:.:::;;X$&&&&&&&&&; .         
    . ..                    .. . ..:::+xXXX$$$$X;.:;x$&$&&&X$&+..         
    . ..                             ....:;+xXX$&&$XxX$$&XX&$$X.          
    . ..                                   ...:;X$&&&&&&&:$&$X+..         
    . ..                                       ...:+&&&&&&&&&X: .         
    . ..                                           ...:+&&&&X:  .         
    . ..                           .   ..    .   .  .  ...... ..          
    ... . . . ..  . .  .  . ...  . . ..  ....  .  .  .    .   .           
                                                                          
```

